#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
rag_core.py — Aozora RAG core (Gemini Embeddings + FAISS)

Features
--------
- Streaming, resumable index build to FAISS (text chunks)
- Exponential backoff with jitter for embedding retries
- Uses sidecar .meta.json from aozora_downloader.py (title/author/card_url)
- Hybrid retrieval (semantic + literal fallback)
- Kanji semantic neighbor suggestion (based on build_char_semantic.py)
- Unified answer() for:
    * free-text queries (normal RAG)
    * single-kanji queries (name helper + usage in works)
"""

from __future__ import annotations
import os
import json
import time
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import faiss
import yaml
from tqdm import tqdm

import google.generativeai as genai


# -------------------------------------------------------------------
# Config & Gemini setup
# -------------------------------------------------------------------
def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Backward-compatible API section
    api_section = cfg.get("api") or {}
    if "gemini_api_key" not in api_section and "gemini_api_key" in cfg:
        api_section["gemini_api_key"] = cfg["gemini_api_key"]
    cfg.setdefault("api", api_section)

    # Backward-compatible model section
    models = cfg.get("models") or {}
    if "embed" not in models and "embed" in cfg:
        models["embed"] = cfg["embed"]
    if "llm" not in models and "llm" in cfg:
        models["llm"] = cfg["llm"]
    cfg.setdefault("models", models)

    # Chunking config
    chunking = cfg.get("chunking") or {}
    chunking.setdefault("max_chars", cfg.get("max_chars", 1200))
    chunking.setdefault("overlap", cfg.get("overlap", 120))
    cfg["chunking"] = chunking

    # Embedding config
    embedding = cfg.get("embedding") or {}
    embedding.setdefault("batch_size", cfg.get("batch_size", 64))
    embedding.setdefault("normalize", cfg.get("normalize", True))
    cfg["embedding"] = embedding

    # Index config
    index = cfg.get("index") or {}
    index.setdefault("dir", cfg.get("dir", "index/faiss"))
    index.setdefault("metadata_jsonl", cfg.get("metadata_jsonl", "metadata.jsonl"))
    index.setdefault("faiss_index", cfg.get("faiss_index", "vectors.faiss"))
    cfg["index"] = index

    # Character semantic index config
    char_index = cfg.get("char_index") or {}
    char_index.setdefault("dir", "index/char_semantic")
    char_index.setdefault("faiss", "faiss_char_semantic.faiss")
    char_index.setdefault("vocab", "char_vocab.jsonl")
    char_index.setdefault("seed_json", "data/kanji_semantic.json")
    cfg["char_index"] = char_index

    return cfg


def setup_gemini(api_key: str | None):
    key = os.getenv("GEMINI_API_KEY") or api_key
    if not key:
        raise RuntimeError(
            "Gemini API key missing. "
            "Set GEMINI_API_KEY or config.yaml api.gemini_api_key"
        )
    genai.configure(api_key=key)


# -------------------------------------------------------------------
# Text I/O and chunking
# -------------------------------------------------------------------
def _read_txt(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8")
    except Exception:
        return fp.read_text(encoding="utf-8", errors="ignore")


def chunk_text(
    text: str,
    max_chars: int = 1200,
    overlap: int = 120,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """Split text into overlapping character-based chunks."""
    chunks: List[str] = []
    spans: List[Tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        j = min(n, i + max_chars)
        chunks.append(text[i:j])
        spans.append((i, j))
        if j == n:
            break
        i = j - overlap
        if i <= 0:
            # Safety guard if overlap >= max_chars
            i = j
    return chunks, spans


# -------------------------------------------------------------------
# Embedding with retry (single + batched)
# -------------------------------------------------------------------
def _retry_embed_once(
    text: str,
    model_name: str,
    max_retries: int = 6,
) -> List[float]:
    """
    Call Gemini embed API with exponential backoff + jitter.
    Return a single embedding vector.
    """
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
                raise RuntimeError(
                    f"Embed response missing 'embedding' key: {str(r)[:200]}"
                )
            return list(emb)

        except Exception:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff with small random jitter
            sleep_s = min(60, 2 ** attempt) + random.uniform(0, 0.5)
            time.sleep(sleep_s)


def embed_texts(
    texts: List[str],
    model_name: str,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """
    Embed a list of texts, in batches, with a progress bar.
    Returns an array of shape [N, D].
    """
    dim_guess = None
    out_blocks: List[np.ndarray] = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        vecs: List[np.ndarray] = []
        for t in batch:
            v = _retry_embed_once(t, model_name)
            if dim_guess is None:
                dim_guess = len(v)
            vecs.append(np.asarray(v, dtype=np.float32))
        block = np.vstack(vecs)
        if normalize:
            norms = np.linalg.norm(block, axis=1, keepdims=True) + 1e-12
            block = block / norms
        out_blocks.append(block)

    if not out_blocks:
        return np.zeros((0, dim_guess or 1), dtype=np.float32)
    return np.vstack(out_blocks)


# -------------------------------------------------------------------
# RAG pipeline
# -------------------------------------------------------------------
class RagPipeline:
    def __init__(self, config_path: str = "config.yaml", data_dir: str = "data"):
        self.cfg = load_config(config_path)
        self.data_dir = data_dir

        setup_gemini(self.cfg["api"].get("gemini_api_key"))

        self.embed_model = self.cfg["models"]["embed"]
        self.llm_model = self.cfg["models"]["llm"]

        self.max_chars = int(self.cfg["chunking"]["max_chars"])
        self.overlap = int(self.cfg["chunking"]["overlap"])

        self.batch_size = int(self.cfg["embedding"]["batch_size"])
        self.normalize = bool(self.cfg["embedding"]["normalize"])

        self.index_dir = self.cfg["index"]["dir"]
        self.meta_jsonl = self.cfg["index"]["metadata_jsonl"]
        self.faiss_index = self.cfg["index"]["faiss_index"]

        # Character semantic index config
        ci = self.cfg.get("char_index", {})
        self.char_index_dir = ci.get("dir", "index/char_semantic")
        self.char_index_faiss = ci.get("faiss", "faiss_char_semantic.faiss")
        self.char_index_vocab = ci.get("vocab", "char_vocab.jsonl")
        self.char_seed_json = Path(ci.get("seed_json", "data/kanji_semantic.json"))

        # Runtime resources (lazy-loaded)
        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.meta_rows: List[Dict[str, Any]] | None = None

        self.char_index: faiss.Index | None = None
        self.char_vocab: List[str] | None = None
        self.char_seed: Dict[str, List[str]] = self._load_char_seed()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_char_seed(self) -> Dict[str, List[str]]:
        """
        Load optional seed JSON with manual semantic descriptions.
        Structure: [{"char": "...", "desc": ["...","..."]}, ...]
        """
        seed: Dict[str, List[str]] = {}
        if self.char_seed_json and self.char_seed_json.exists():
            try:
                data = json.loads(
                    self.char_seed_json.read_text(encoding="utf-8")
                )
                if isinstance(data, list):
                    for row in data:
                        ch = row.get("char")
                        desc = row.get("desc") or []
                        if ch and isinstance(desc, list):
                            seed[ch] = desc
            except Exception:
                pass
        return seed

    def _scan_and_chunk(
        self,
        data_dir: str,
    ) -> Tuple[List[str], List[Tuple[int, int]], List[Dict[str, Any]]]:
        """
        Scan all .txt files and chunk them.

        Returns:
            texts: list of chunk strings
            spans: list of (start, end) offsets within each source text
            meta : per-chunk metadata (path/title/author/card_url)
        """
        texts: List[str] = []
        spans: List[Tuple[int, int]] = []
        meta: List[Dict[str, Any]] = []

        for fp in sorted(Path(data_dir).glob("*.txt")):
            doc = _read_txt(fp)
            doc_chunks, doc_spans = chunk_text(doc, self.max_chars, self.overlap)

            # Optional sidecar metadata written by aozora_downloader
            side = fp.with_suffix(".meta.json")
            title = author = card_url = ""
            if side.exists():
                try:
                    j = json.loads(side.read_text(encoding="utf-8"))
                    title = j.get("title", "") or title
                    author = j.get("author", "") or author
                    card_url = j.get("card_url", "") or card_url
                except Exception:
                    pass

            for (s, e), ch in zip(doc_spans, doc_chunks):
                texts.append(ch)
                spans.append((s, e))
                meta.append(
                    {
                        "path": str(fp),
                        "title": title,
                        "author": author,
                        "card_url": card_url,
                    }
                )

        return texts, spans, meta

    # ------------------------------------------------------------------
    # Streaming FAISS build (resumable)
    # ------------------------------------------------------------------
    def build_index(self):
        """
        Build or resume the main FAISS index over text chunks.

        The index is constructed in a streaming way:
        - metadata.jsonl is appended line-by-line
        - vectors are added to FAISS incrementally
        - writing to a temporary FAISS file, then atomically renamed
        """
        t0 = time.time()
        texts, spans, meta = self._scan_and_chunk(self.data_dir)
        print(f"[⏱] Scanning & chunking: {time.time() - t0:.2f}s")

        out_dir = Path(self.index_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        meta_path = out_dir / self.meta_jsonl
        faiss_path = out_dir / self.faiss_index
        tmp_faiss_path = out_dir / (self.faiss_index + ".building")

        # How many chunks are already persisted in metadata.jsonl?
        processed = 0
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as f:
                for _ in f:
                    processed += 1

        # Load existing temporary FAISS index if available
        index = None
        dim = None
        if tmp_faiss_path.exists():
            index = faiss.read_index(str(tmp_faiss_path))
            dim = index.d
        elif faiss_path.exists() and processed == len(spans):
            print("Index already complete; nothing to do.")
            return self, spans

        if index is None:
            # Probe a single vector to get the dimension
            probe = _retry_embed_once(texts[0], self.embed_model)
            dim = len(probe)
            index = faiss.IndexFlatIP(dim)

            # If metadata existed but FAISS index did not, replay vectors
            if processed > 0:
                print(f"Replaying first {processed} vectors into FAISS (resume)...")
                replay_texts = texts[:processed]
                vecs = embed_texts(
                    replay_texts,
                    self.embed_model,
                    self.batch_size,
                    normalize=self.normalize,
                )
                index.add(vecs)

        t1 = time.time()
        batch_vecs: List[np.ndarray] = []
        batch_meta: List[Dict[str, Any]] = []
        checkpoint_every = 5000
        added_since_ckpt = 0

        meta_f = None
        if processed < len(spans):
            meta_f = (out_dir / self.meta_jsonl).open("a", encoding="utf-8")

        try:
            for i in tqdm(
                range(processed, len(spans)), desc="Building index (streaming)"
            ):
                m = {
                    "path": meta[i]["path"],
                    "title": meta[i].get("title", ""),
                    "author": meta[i].get("author", ""),
                    "card_url": meta[i].get("card_url", ""),
                    "start": spans[i][0],
                    "end": spans[i][1],
                }
                v = _retry_embed_once(texts[i], self.embed_model)
                v = np.asarray(v, dtype=np.float32)[None, :]
                if self.normalize:
                    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

                batch_vecs.append(v)
                batch_meta.append(m)

                if len(batch_vecs) >= self.batch_size:
                    block = np.vstack(batch_vecs)
                    index.add(block)
                    for row in batch_meta:
                        meta_f.write(
                            json.dumps(row, ensure_ascii=False) + "\n"
                        )
                    meta_f.flush()
                    added_since_ckpt += block.shape[0]
                    batch_vecs.clear()
                    batch_meta.clear()

                if added_since_ckpt >= checkpoint_every:
                    faiss.write_index(index, str(tmp_faiss_path))
                    added_since_ckpt = 0

            # Flush remainder
            if batch_vecs:
                block = np.vstack(batch_vecs)
                index.add(block)
                for row in batch_meta:
                    meta_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                meta_f.flush()
                batch_vecs.clear()
                batch_meta.clear()

        finally:
            if meta_f:
                meta_f.close()

        # Finalize (atomic replace)
        faiss.write_index(index, str(tmp_faiss_path))
        tmp_faiss_path.replace(faiss_path)

        print(
            f"[⏱] Embedding {len(spans)} chunks using {self.embed_model}: "
            f"{time.time() - t1:.2f}s"
        )
        return self, spans

    # ------------------------------------------------------------------
    # Loading index / metadata / char index
    # ------------------------------------------------------------------
    def _ensure_loaded(self):
        """Load main FAISS index + metadata.jsonl if not yet loaded."""
        if self.index is None:
            faiss_path = Path(self.index_dir) / self.faiss_index
            if not faiss_path.exists():
                raise RuntimeError(
                    "FAISS index not found. Run `python rag_core.py --build` first."
                )
            self.index = faiss.read_index(str(faiss_path))

        if self.meta_rows is None:
            meta_path = Path(self.index_dir) / self.meta_jsonl
            rows: List[Dict[str, Any]] = []
            with meta_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        # Skip malformed lines
                        continue
            self.meta_rows = rows

    def _ensure_char_index_loaded(self):
        """Load character semantic FAISS index and vocabulary."""
        if self.char_index is not None and self.char_vocab is not None:
            return

        dir_ = Path(self.char_index_dir)
        idx_path = dir_ / self.char_index_faiss
        vocab_path = dir_ / self.char_index_vocab

        if not idx_path.exists() or not vocab_path.exists():
            raise RuntimeError(
                "Character semantic index not found. "
                "Run `python build_char_semantic.py` first."
            )

        self.char_index = faiss.read_index(str(idx_path))

        chars: List[str] = []
        with vocab_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    ch = obj.get("char")
                    if ch:
                        chars.append(ch)
                except Exception:
                    continue
        self.char_vocab = chars

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------
    def _norm(self, s: str) -> str:
        import unicodedata

        return unicodedata.normalize("NFKC", s or "")

    def _get_snippet_cached(self, cache: dict, m: dict) -> str:
        """Read snippet [start:end] with a small per-path cache."""
        p = m["path"]
        doc = cache.get(p)
        if doc is None:
            try:
                doc = Path(p).read_text(encoding="utf-8")
            except Exception:
                doc = Path(p).read_text(encoding="utf-8", errors="ignore")
            cache[p] = doc
        s, e = int(m.get("start", 0)), int(m.get("end", 0))
        return doc[s:e]

    # ------------------------------------------------------------------
    # Hybrid retrieve (semantic + literal + corpus-wide literal fallback)
    # ------------------------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        pool_k: int = 120,
        literal_first: bool = True,
        literal_boost: float = 0.75,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid retrieve with a global literal fallback.

        1) Semantic retrieval to get a pool of candidates.
        2) Literal boosts on that pool.
        3) If there is no literal hit in the pool, scan the entire corpus
           for literal hits and prepend them.
        """
        self._ensure_loaded()
        q_raw = query
        query = self._norm(query)

        # 1) Semantic retrieval
        qv = _retry_embed_once(query, self.embed_model)
        qv = np.asarray(qv, dtype=np.float32)[None, :]
        if self.normalize:
            qv = qv / (np.linalg.norm(qv, axis=1, keepdims=True) + 1e-12)

        k = min(max(top_k, 5), pool_k, self.index.ntotal)
        sims, idx = self.index.search(qv, k)
        idx, sims = idx[0], sims[0]

        candidates: List[Dict[str, Any]] = []
        for i, s in zip(idx, sims):
            if i < 0:
                continue
            m = dict(self.meta_rows[i])
            m["score_sem"] = float(s)
            candidates.append(m)

        # 2) literal re-weighting on semantic pool
        cache: Dict[str, str] = {}
        for m in candidates:
            snip = self._get_snippet_cached(cache, m)
            hits = self._norm(snip).count(query)
            m["literal_hits"] = hits
            bonus = (literal_boost if hits > 0 else 0.0) + 0.05 * hits
            m["score_mix"] = m["score_sem"] + bonus

        any_hit = any(m.get("literal_hits", 0) > 0 for m in candidates)

        # 3) global literal fallback if there was no literal hit
        if not any_hit and query:
            literal_matches: List[Dict[str, Any]] = []

            # Corpus-wide scan is fine for ~k thousands of chunks
            for m_full in self.meta_rows:
                snip = self._get_snippet_cached(cache, m_full)
                if self._norm(snip).count(query) > 0:
                    mf = dict(m_full)
                    mf["literal_hits"] = 1
                    mf["score_sem"] = 0.0
                    mf["score_mix"] = 1.0  # Put these at the very top
                    literal_matches.append(mf)
                    if len(literal_matches) >= max(top_k * 3, 30):
                        break

            if literal_matches:
                ranked = literal_matches + sorted(
                    candidates, key=lambda x: x["score_mix"], reverse=True
                )
            else:
                ranked = sorted(
                    candidates, key=lambda x: x["score_mix"], reverse=True
                )
        else:
            # Normal merging of literal hits + non-hits
            if literal_first:
                winners = [m for m in candidates if m.get("literal_hits", 0) > 0]
                losers = [m for m in candidates if m.get("literal_hits", 0) == 0]
                winners.sort(key=lambda x: x["score_mix"], reverse=True)
                losers.sort(key=lambda x: x["score_mix"], reverse=True)
                ranked = winners + losers
            else:
                ranked = sorted(
                    candidates, key=lambda x: x["score_mix"], reverse=True
                )

        out: List[Dict[str, Any]] = []
        for m in ranked[:top_k]:
            m["score"] = float(m.get("score_mix", m.get("score_sem", 0.0)))
            out.append(m)
        return out

    # ------------------------------------------------------------------
    # Normal RAG answer (free text)
    # ------------------------------------------------------------------
    def answer(self, query: str, top_k: int = 5) -> str:
        """
        Standard RAG answer for free-text queries.
        """
        ctxs = self.retrieve(query, top_k=top_k)
        pieces: List[str] = []
        for m in ctxs:
            try:
                doc = Path(m["path"]).read_text(encoding="utf-8")
            except Exception:
                doc = Path(m["path"]).read_text(encoding="utf-8", errors="ignore")
            s, e = int(m.get("start", 0)), int(m.get("end", 0))
            snippet = doc[s:e]
            src = m.get("title") or Path(m["path"]).name
            pieces.append(f"[{src}] {snippet}")

        prompt = (
            "You are a helpful assistant for Japanese literature. "
            "Given the user's query and retrieved snippets, answer the question, "
            "explain briefly, and cite the sources (title/author). "
            "If the query is not answered by the snippets, say so briefly.\n\n"
            f"Query: {query}\n\n"
            "Snippets:\n" + "\n---\n".join(pieces[:top_k])
        )

        model = genai.GenerativeModel(self.llm_model)
        resp = model.generate_content(prompt)
        return resp.text if hasattr(resp, "text") and resp.text else str(resp)

    # ------------------------------------------------------------------
    # Character semantic helpers
    # ------------------------------------------------------------------
    def _embed_char_for_semantic(self, ch: str) -> np.ndarray:
        """
        Embed a short English prompt describing the kanji, instead of
        just the raw character. This gives much better semantic structure.
        """
        prompt = (
            f"Explain briefly the core meanings and typical nuances of the "
            f"kanji '{ch}' in Japanese (one short sentence)."
        )
        v = _retry_embed_once(prompt, self.embed_model)
        v = np.asarray(v, dtype=np.float32)[None, :]
        if self.normalize:
            v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        return v

    def suggest_similar_chars(self, ch: str, topn: int = 5) -> List[Dict[str, Any]]:
        """
        Query the character semantic FAISS index and return top-N
        semantically similar kanji, with optional descriptions from seed JSON.
        """
        self._ensure_char_index_loaded()
        qv = self._embed_char_for_semantic(ch)

        k = min(topn, self.char_index.ntotal)
        sims, idx = self.char_index.search(qv, k)

        out: List[Dict[str, Any]] = []
        for i, score in zip(idx[0], sims[0]):
            if i < 0:
                continue
            cand = self.char_vocab[i]
            desc = self.char_seed.get(cand)
            out.append(
                {
                    "char": cand,
                    "score": float(score),
                    "desc": desc,
                }
            )
        return out

    def find_char_occurrences(
        self,
        ch: str,
        max_docs: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Scan metadata.jsonl and collect which documents contain this
        kanji and how many times it appears (approximate, per chunk).
        """
        self._ensure_loaded()
        cache: Dict[str, str] = {}
        doc_hits: Dict[str, Dict[str, Any]] = {}

        for m in self.meta_rows:
            path = m["path"]
            snip = self._get_snippet_cached(cache, m)
            if ch in snip:
                rec = doc_hits.get(path)
                if rec is None:
                    rec = {
                        "path": path,
                        "title": m.get("title", ""),
                        "author": m.get("author", ""),
                        "count": 0,
                    }
                    doc_hits[path] = rec
                rec["count"] += snip.count(ch)

        docs = sorted(
            doc_hits.values(),
            key=lambda x: x["count"],
            reverse=True,
        )
        return docs[:max_docs]

    # ------------------------------------------------------------------
    # Kanji helpers for unified answer / char dialog
    # ------------------------------------------------------------------
    def _build_kanji_examples(
        self,
        ch: str,
        max_docs: int,
    ) -> List[Dict[str, Any]]:
        """
        Collect a few works where the kanji appears and extract
        short snippets for LLM explanation.
        """
        occ_docs = self.find_char_occurrences(ch, max_docs=max_docs)
        examples: List[Dict[str, Any]] = []

        for d in occ_docs:
            path = d["path"]
            title = d.get("title") or Path(path).stem
            author = d.get("author") or ""

            try:
                doc = Path(path).read_text(encoding="utf-8")
            except Exception:
                doc = Path(path).read_text(encoding="utf-8", errors="ignore")

            pos = doc.find(ch)
            if pos == -1:
                start = 0
                end = min(len(doc), 80)
            else:
                start = max(0, pos - 40)
                end = min(len(doc), pos + 40)

            snippet = doc[start:end].replace("\n", "")
            examples.append(
                {
                    "title": title,
                    "author": author,
                    "path": path,
                    "count": d["count"],
                    "snippet": snippet,
                }
            )

        return examples

    def _llm_explain_kanji_with_examples(
        self,
        ch: str,
        examples: List[Dict[str, Any]],
    ) -> str:
        """
        Ask the LLM to explain the kanji meaning + how it is used in each work.
        All works in `examples` are guaranteed to actually contain the kanji.
        """
        model = genai.GenerativeModel(self.llm_model)

        lines: List[str] = []
        lines.append(
            "You are a helpful assistant for Japanese literature and kanji usage."
        )
        lines.append(
            f"Target kanji: '{ch}'. "
            "Explain it for a Japanese reader who is considering using this kanji in a personal name."
        )
        lines.append(
            "Output the answer **entirely in Japanese**.\n"
            "構成は次のようにしてください（英語の説明文はそのまま書かないでください）：\n"
            "1. 「{kanji}」の基本的な意味\n"
            "   - 箇条書きで2〜4個ほど、字義とニュアンスを説明する。\n"
            "2. 作品中での用例\n"
            "   - 『タイトル』― 著者：上に示した抜粋（SNIPPET）を参考に、\n"
            "     この漢字がどのような意味・雰囲気で使われているかを短く説明する。\n"
            "※ 上の番号や説明文はガイドラインなので、そのままコピーせず、\n"
            "   自然な日本語の見出しと文章として書いてください。"
        )
        lines.append(
            "IMPORTANT:\n"
            f"- For every work listed below, you must assume the kanji '{ch}' DOES appear in that work.\n"
            "- Do NOT say that the kanji does not appear in any of these works.\n"
            "- If the snippet is short or the context is unclear, you may say that the nuance is not fully clear, "
            "but still treat the kanji as present in the work."
        )
        lines.append("Works and snippets:")

        for ex in examples:
            meta = ex["title"]
            if ex["author"]:
                meta += f" — {ex['author']}"
            lines.append(
                f"\n[WORK] {meta}\n"
                f"[COUNT] approximately {ex['count']} occurrences\n"
                f"[SNIPPET]\n{ex['snippet']}\n"
            )

        prompt = "\n".join(lines)
        resp = model.generate_content(prompt)
        base_text = resp.text if getattr(resp, "text", None) else str(resp)

        # Append a clean source list
        src_lines = ["\nSources:"]
        for ex in examples:
            meta = ex["title"]
            if ex["author"]:
                meta += f" — {ex['author']}"
            src_lines.append(
                f"- {meta} (approximately {ex['count']} occurrences)"
            )
        return base_text + "\n" + "\n".join(src_lines)

    
    def _llm_plain_kanji_explanation(self, ch: str) -> str:
        """
        Simple fallback: explain one kanji without any neighbors or corpus context.
        Used when we cannot find reasonable semantic neighbors.
        """
        model = genai.GenerativeModel(self.llm_model)
        prompt = (
            "You are a helpful assistant for Japanese kanji.\n"
            f"Explain the typical meanings, nuances, and common name usage of the kanji '{ch}' in Japanese.\n"
            "Focus on:\n"
            "- Core meanings (2–4 bullet points)\n"
            "- Emotional / stylistic nuance (1–2 sentences)\n"
            "- Whether it is commonly used in personal names, and if so, what kind of image it gives.\n"
            "Answer concisely in Japanese, using Markdown bullet lists where appropriate."
        )
        resp = model.generate_content(prompt)
        return resp.text if getattr(resp, "text", None) else str(resp)

    def _kanji_no_occurrence_dialog(
        self,
        ch: str,
        top_neighbors: int = 5,
    ) -> str:
        """
        Used when the kanji does NOT appear in the corpus.
        Try to propose a few semantically similar kanji. If neighbors
        look unreliable (very low scores), fall back to a plain
        LLM explanation of the target kanji only.
        """
        neighbors = self.suggest_similar_chars(ch, top_neighbors)

        # Filter by a simple similarity threshold
        min_score = 0.25
        filtered = [
            nb for nb in neighbors if nb.get("score", 0.0) >= min_score
        ]

        # If we have no reasonably close neighbors, just explain the kanji itself.
        if not filtered:
            return self._llm_plain_kanji_explanation(ch)

        # Build a prompt that describes the main kanji + its neighbors.
        model = genai.GenerativeModel(self.llm_model)
        lines: List[str] = []
        lines.append(
            "You are a helpful assistant for Japanese kanji and name selection."
        )
        lines.append(
            f"Target kanji: '{ch}'. It does not appear in the current corpus, "
            "but we have a few semantically related kanji candidates."
        )
        lines.append(
            "For the target kanji and each candidate, explain:\n"
            "- Core meanings / nuances\n"
            "- Typical image if used in a personal name\n"
            "Do NOT judge whether the relation is weak or strong. "
            "Just describe each kanji's meaning and nuance in a positive, neutral way."
        )
        lines.append("\nList of candidate kanji:")
        for nb in filtered:
            desc_list = nb.get("desc") or []
            desc_str = "; ".join(desc_list) if desc_list else ""
            lines.append(
                f"- {nb['char']} (similarity score ≈ {nb['score']:.3f})"
                + (f" | seed hints: {desc_str}" if desc_str else "")
            )

        prompt = "\n".join(lines)
        resp = model.generate_content(prompt)
        return resp.text if getattr(resp, "text", None) else str(resp)

    # ------------------------------------------------------------------
    # Character dialog / unified kanji answer
    # ------------------------------------------------------------------
    def char_dialog(
        self,
        ch: str,
        top_neighbors: int = 5,
        max_docs_per_char: int = 3,
    ) -> str:
        """
        High-level flow for kanji queries.

        - If the kanji appears in the corpus:
            → ask LLM to explain meaning + usage in each work.
        - If not:
            → suggest semantically similar kanji; if no good neighbor,
              fall back to a plain LLM explanation of the kanji itself.
        """
        ch = (ch or "").strip()
        if not ch:
            return "Please provide a non-empty character."
        if len(ch) > 1:
            # Still allow multi-character input, but we focus on the first char.
            ch = ch[0]

        # 1) Check direct occurrences in the corpus
        examples = self._build_kanji_examples(ch, max_docs=max_docs_per_char)
        if examples:
            return self._llm_explain_kanji_with_examples(ch, examples)

        # 2) No occurrences → neighbor-based dialog (with safe fallback).
        return self._kanji_no_occurrence_dialog(ch, top_neighbors=top_neighbors)

    # ------------------------------------------------------------------
    # Unified answer for single kanji (used by CLI and chat)
    # ------------------------------------------------------------------
    
    def answer_unified(self, query: str, top_k: int = 5) -> str:
        """
        Human-friendly explanation for a single kanji, aimed at name selection use.
        Style:
          - Mainly Japanese, with some English headings / phrases allowed
          - Warm, natural, not a rigid template
          - Same tone whether the kanji appears in the corpus or not
        """
        q = (query or "").strip()
        if not q:
            return "漢字が入力されていません。１文字だけ入れてみてくださいね。"

        # 1) Retrieve normal text contexts (for compounds like 瑩徹, 瑩朗, etc.)
        ctxs = self.retrieve(q, top_k=top_k)
        snippet_blocks: list[str] = []
        for m in ctxs:
            try:
                doc = Path(m["path"]).read_text(encoding="utf-8")
            except Exception:
                doc = Path(m["path"]).read_text(encoding="utf-8", errors="ignore")
            s, e = int(m.get("start", 0)), int(m.get("end", 0))
            snippet = doc[s:e]
            title = m.get("title") or Path(m["path"]).name
            author = m.get("author") or ""
            header = f"{title} — {author}" if author else title
            snippet_blocks.append(f"[{header}]\n{snippet}")

        corpus_block = "\n\n---\n\n".join(snippet_blocks) if snippet_blocks else "(no snippets found)"

        # 2) Get occurrence summary + semantic neighbors from our char dialog helper
        #    （這裡只是拿現成的資料，讓 LLM 重新整理成好讀的說明）
        char_view = self.char_dialog(q, top_neighbors=min(8, top_k), max_docs_per_char=3)

        # 3) Prompt Gemini with style instructions (不固定格式，只定調性）
        prompt = f"""
You are helping a Japanese parent who is thinking about kanji for a personal name.

Target kanji 
(include the readings, use the format 「◯◯（かな / romaji）」
if there are more than one readings then must list all the readings and use the format「◯◯（かな1 / romaji1　、　「◯◯（かな2 / romaji2）」）」): 「{q}」
Please include **all the possible readings**「◯◯（かな / romaji）」for this kanji at the beginning.

You are given:

[1] Aozora Bunko occurrence summary:
{char_view}

[2] Text snippets:
{corpus_block}

Please write a warm, natural Japanese explanation (not a template) with the rules below.

========================
🌸 STYLE
========================
- Gentle, soft, conversational Japanese.
- No rigid structure; write 2–4 natural paragraphs.
- Occasional small English headings like “Meaning”, “Name impression” are OK, but avoid hard, technical English.
- Do NOT say “core meaning”, “semantic nuance” etc. Use normal Japanese wording.
- No bullet points unless they feel natural.

========================
🌸 REQUIRED CONTENT
========================

1. Begin with a gentle opening such as:
   「はい、こちらの漢字『{q}』について少し丁寧に説明してみますね。」

2. Explain the meaning and emotional impression of the kanji in natural Japanese.

3. If Aozora Bunko has examples:
   - Clearly mention *author + work name* such as:
     「夏目漱石『吾輩は猫である』では〜」
     「◯◯『□□』の中で〜」
   - Briefly explain how the kanji is used in that exact sentence or compound.

4. If there are no occurrences:
   - Gently say so in Japanese without sounding negative.

5. If suggesting related kanji:
   - Provide all possible readings (かな + romaji), e.g.:
       「◯◯（かな / romaji）」など
   - Only introduce 1–3 kanji that genuinely fit the name-image.
   - Do NOT use specific fixed examples in the prompt.

6. Provide a soft “Name impression” section describing the feeling or hope conveyed when using this kanji in a name.

========================
🌸 SOURCES (MANDATORY)
========================
At the end, output a section titled **Sources**, listing the detected occurrences exactly as:

Sources:
・吾輩は猫である — 夏目漱石（2 occurrences）
・薤露行 — 夏目漱石（1 occurrences）

If none:
Sources:
・Aozora Bunko: No occurrences found

Do NOT skip this section under any circumstances.

Write only the final answer.
"""



        model = genai.GenerativeModel(self.llm_model)
        resp = model.generate_content(prompt)
        return resp.text if hasattr(resp, "text") and resp.text else str(resp)


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Aozora RAG core")
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--data", type=str, default="data")

    ap.add_argument(
        "--build",
        action="store_true",
        help="Build (or resume) the main FAISS index",
    )
    ap.add_argument(
        "--query",
        type=str,
        default=None,
        help="Free-text query or a single kanji for kanji mode",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=5,
        help="Number of contexts / neighbors to use",
    )
    ap.add_argument(
        "--char-dialog",
        type=str,
        default=None,
        help=(
            "Inspect a kanji: if present, explain usage in works; "
            "if not, suggest semantically similar characters."
        ),
    )

    args = ap.parse_args()
    rag = RagPipeline(config_path=args.config, data_dir=args.data)

    if args.build:
        rag.build_index()
        return

    if args.char_dialog:
        out = rag.char_dialog(
            args.char_dialog,
            top_neighbors=args.topk,
            max_docs_per_char=args.topk,
        )
        print(out)
        return

    if args.query:
        ans = rag.answer_unified(args.query, top_k=args.topk)
        print(ans)
        return

    ap.print_help()


if __name__ == "__main__":
    main()
