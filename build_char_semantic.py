#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_char_semantic.py

Two-stage pipeline to build a *semantic* kanji index:

1) --defs
   - Scan data/*.txt and collect kanji in a given freq range.
   - For each kanji, call an LLM to get 3–5 short English
     keywords that describe its meaning/nuance.
   - Append results to data/kanji_semantic_all.jsonl (resumable).

2) --index
   - Read data/kanji_semantic_all.jsonl.
   - Embed the joined English keywords with text-embedding-004.
   - Write:
       index/char_semantic/faiss_char_semantic.faiss
       index/char_semantic/char_vocab.jsonl
"""

from __future__ import annotations
import os
import re
import json
import time
import random
from pathlib import Path
from collections import Counter, deque
from typing import List, Dict, Any

import asyncio
import numpy as np
import faiss
import yaml
from tqdm import tqdm
import google.generativeai as genai

# --------------------------------------------------------------------
# Paths / constants
# --------------------------------------------------------------------
CONFIG_PATH = "config.yaml"
DATA_DIR = Path("data")
OUT_DIR = Path("index/char_semantic")
OUT_DIR.mkdir(parents=True, exist_ok=True)

KANJI_DEFS_JSONL = DATA_DIR / "kanji_semantic_all.jsonl"  # auto-generated, resumable
CHAR_VOCAB_JSONL = OUT_DIR / "char_vocab.jsonl"
FAISS_PATH = OUT_DIR / "faiss_char_semantic.faiss"

KANJI_RX = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")

# Concurrency / rate limiting (safe vs your 2000 RPM limit)
MAX_CONCURRENCY = 20          # how many LLM calls in flight at once
MAX_LLM_RPM = 600             # max requests per minute (global), safety cap
MAX_RETRIES = 6               # retries per LLM call


# --------------------------------------------------------------------
# Config & Gemini setup
# --------------------------------------------------------------------
def load_cfg() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    api = cfg.get("api") or {}
    key = os.getenv("GEMINI_API_KEY") or api.get("gemini_api_key")
    if not key:
        raise RuntimeError("Missing Gemini API key. Set GEMINI_API_KEY or api.gemini_api_key in config.yaml.")

    genai.configure(api_key=key)

    models = cfg.get("models") or {}
    embed_model = models.get("embed", "text-embedding-004")
    # Any text model that supports generate_content is fine here
    llm_model = models.get("llm", "gemini-2.0-flash")
    return {"embed": embed_model, "llm": llm_model}


# --------------------------------------------------------------------
# Simple rate limiter for async use
# --------------------------------------------------------------------
class RateLimiter:
    """
    Very simple async rate limiter: at most `max_calls` within `period` seconds.
    It keeps a deque of timestamps and sleeps when necessary.
    """

    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait until it is safe to perform the next call."""
        while True:
            async with self._lock:
                now = time.time()
                # Drop timestamps that are outside of the window
                while self._timestamps and now - self._timestamps[0] > self.period:
                    self._timestamps.popleft()

                if len(self._timestamps) < self.max_calls:
                    # We are under the limit; record this call and proceed
                    self._timestamps.append(now)
                    return

                # Otherwise, compute how long to wait
                wait_for = self.period - (now - self._timestamps[0]) + 0.01
            # Sleep outside the lock
            await asyncio.sleep(max(wait_for, 0.01))


# --------------------------------------------------------------------
# Low-level embedding helper (synchronous, used only in --index)
# --------------------------------------------------------------------
def _retry_embed(text: str, model_name: str, max_retries: int = MAX_RETRIES) -> List[float]:
    """Call Gemini embedding with exponential backoff + jitter."""
    for attempt in range(max_retries):
        try:
            r = genai.embed_content(model=model_name, content=text)
            emb = None
            if isinstance(r, dict):
                emb = r.get("embedding")
                if isinstance(emb, dict):
                    emb = emb.get("values")
            else:
                emb = getattr(r, "embedding", None)
                if hasattr(emb, "values"):
                    emb = emb.values
            if emb is None:
                raise RuntimeError(f"Embed response missing 'embedding'. Preview: {str(r)[:200]}")
            return list(emb)
        except Exception:
            if attempt == max_retries - 1:
                raise
            sleep_s = min(60, 2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)


# --------------------------------------------------------------------
# Collect kanji from corpus
# --------------------------------------------------------------------
def collect_chars_from_corpus(
    data_dir: Path = DATA_DIR,
    min_freq: int = 1,
    max_freq: int = 200,
) -> tuple[list[str], Dict[str, int]]:
    """
    Scan data/*.txt and count kanji occurrences.
    Keep characters whose frequency is in [min_freq, max_freq].
    """
    cnt: Counter[str] = Counter()
    for fp in data_dir.glob("*.txt"):
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        for ch in KANJI_RX.findall(txt):
            cnt[ch] += 1

    chars = [c for c, n in cnt.items() if min_freq <= n <= max_freq]
    chars.sort()
    return chars, {c: int(cnt[c]) for c in chars}


# --------------------------------------------------------------------
# Definitions JSONL handling
# --------------------------------------------------------------------
def load_existing_definitions() -> Dict[str, List[str]]:
    """
    Read kanji_semantic_all.jsonl if it exists and return {char: desc_list}.
    """
    result: Dict[str, List[str]] = {}
    if not KANJI_DEFS_JSONL.exists():
        return result
    with KANJI_DEFS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ch = obj.get("char")
            desc = obj.get("desc") or []
            if isinstance(ch, str) and ch and isinstance(desc, list):
                result[ch] = [str(x) for x in desc]
    return result


def parse_keywords(raw: str) -> List[str]:
    """
    Parse a comma-separated keyword line into a clean list.
    If parsing fails, fall back to using the whole string.
    """
    if not raw:
        return []
    raw = raw.strip()
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    return parts or [raw]


# --------------------------------------------------------------------
# Async LLM definition generation
# --------------------------------------------------------------------
def _llm_call_sync(model: genai.GenerativeModel, prompt: str) -> str:
    """
    Synchronous LLM call with retry. This is meant to be run
    inside a thread via asyncio.to_thread().
    """
    for attempt in range(MAX_RETRIES):
        try:
            resp = model.generate_content(prompt)
            text = getattr(resp, "text", None) or str(resp)
            return text.strip()
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            sleep_s = min(60, 2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)


async def _generate_one_char(
    ch: str,
    model: genai.GenerativeModel,
    limiter: RateLimiter,
    sem: asyncio.Semaphore,
) -> Dict[str, Any] | None:
    """
    Async worker for a single kanji:
    - respects semaphore (concurrency)
    - respects rate limiter
    - runs blocking LLM call in a thread
    """
    prompt = (
        f"Give 3–5 short English keywords that capture the core meanings "
        f"and typical nuances of the Japanese kanji '{ch}'. "
        "Output only the keywords separated by commas, without any extra text."
    )

    async with sem:
        await limiter.acquire()
        try:
            raw = await asyncio.to_thread(_llm_call_sync, model, prompt)
            desc_list = parse_keywords(raw)
            return {"char": ch, "desc": desc_list}
        except Exception as e:
            print(f"[WARN] failed for kanji '{ch}': {e}")
            return None


async def _generate_defs_async(
    min_freq: int = 1,
    max_freq: int = 200,
):
    """
    Async driver: generate definitions for all kanji in the given
    frequency range that are not already in kanji_semantic_all.jsonl.
    """
    cfg = load_cfg()
    llm_model_name = cfg["llm"]
    model = genai.GenerativeModel(llm_model_name)

    chars, freq = collect_chars_from_corpus(DATA_DIR, min_freq=min_freq, max_freq=max_freq)
    existing = load_existing_definitions()
    already = set(existing.keys())

    to_do = [c for c in chars if c not in already]

    if not to_do:
        print("All kanji in the specified freq range already have definitions. Nothing to do.")
        return

    print(f"Total kanji in range: {len(chars)}  (min_freq={min_freq}, max_freq={max_freq})")
    print(f"Already defined: {len(already)}")
    print(f"To generate in this run: {len(to_do)}")
    print(f"Using concurrency={MAX_CONCURRENCY}, max_rpm={MAX_LLM_RPM}")

    limiter = RateLimiter(max_calls=MAX_LLM_RPM, period=60.0)
    sem = asyncio.Semaphore(MAX_CONCURRENCY)

    tasks = [
        _generate_one_char(ch, model, limiter, sem)
        for ch in to_do
    ]

    results: List[Dict[str, Any]] = []
    # Use tqdm with asyncio.as_completed to show progress
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating kanji definitions"):
        res = await coro
        if res is not None:
            results.append(res)

    if not results:
        print("No new definitions generated.")
        return

    # Append to JSONL (resumable)
    with KANJI_DEFS_JSONL.open("a", encoding="utf-8") as f_out:
        for obj in results:
            f_out.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"Appended {len(results)} new definitions to {KANJI_DEFS_JSONL}")


def generate_defs(min_freq: int = 1, max_freq: int = 200):
    """
    Public entry point for --defs.
    Wraps the async logic with asyncio.run().
    """
    asyncio.run(_generate_defs_async(min_freq=min_freq, max_freq=max_freq))


# --------------------------------------------------------------------
# Build FAISS index from definitions
# --------------------------------------------------------------------
def load_definitions_for_index() -> List[Dict[str, Any]]:
    """
    Load all entries from kanji_semantic_all.jsonl.
    Each entry is {"char": ..., "desc": [..]}.
    """
    rows: List[Dict[str, Any]] = []
    if not KANJI_DEFS_JSONL.exists():
        raise RuntimeError(f"No definitions file found: {KANJI_DEFS_JSONL}")
    with KANJI_DEFS_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ch = obj.get("char")
            desc = obj.get("desc") or []
            if not isinstance(ch, str) or not ch:
                continue
            if not isinstance(desc, list):
                desc = [str(desc)]
            desc = [str(x) for x in desc if str(x).strip()]
            if not desc:
                continue
            rows.append({"char": ch, "desc": desc})
    if not rows:
        raise RuntimeError(f"No usable rows in {KANJI_DEFS_JSONL}")
    return rows


def build_index_from_defs(min_freq: int = 1, max_freq: int = 200):
    """
    Embed the English keyword list for each kanji and build FAISS index
    + char_vocab.jsonl.
    """
    cfg = load_cfg()
    embed_model = cfg["embed"]

    rows = load_definitions_for_index()
    _, freq_map = collect_chars_from_corpus(DATA_DIR, min_freq=min_freq, max_freq=max_freq)

    print(f"Loaded {len(rows)} kanji definitions.")
    texts = ["; ".join(r["desc"]) for r in rows]

    vecs: List[np.ndarray] = []
    for t in tqdm(texts, desc="Embedding kanji meanings"):
        v = _retry_embed(t, embed_model)
        vecs.append(np.asarray(v, dtype=np.float32))
    mat = np.vstack(vecs)
    mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

    dim = mat.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(mat)

    faiss.write_index(index, str(FAISS_PATH))

    with CHAR_VOCAB_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            ch = row["char"]
            freq = int(freq_map.get(ch, 0))
            out = {
                "char": ch,
                "freq": freq,
                "desc": row["desc"],
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Built char semantic index: {FAISS_PATH}")
    print(f"Vocab written to: {CHAR_VOCAB_JSONL}")
    print(f"Total chars indexed: {index.ntotal}")


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Build semantic kanji index (two-stage, async).")
    ap.add_argument(
        "--defs",
        action="store_true",
        help="Generate kanji semantic keyword definitions (resumable, async).",
    )
    ap.add_argument(
        "--index",
        action="store_true",
        help="Build FAISS index from existing definitions.",
    )
    ap.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Minimum frequency in corpus to include a kanji (default=1).",
    )
    ap.add_argument(
        "--max-freq",
        type=int,
        default=200,
        help="Maximum frequency in corpus to include a kanji (default=200).",
    )
    args = ap.parse_args()

    if not args.defs and not args.index:
        ap.print_help()
        return

    if args.defs:
        generate_defs(min_freq=args.min_freq, max_freq=args.max_freq)

    if args.index:
        build_index_from_defs(min_freq=args.min_freq, max_freq=args.max_freq)


if __name__ == "__main__":
    main()
