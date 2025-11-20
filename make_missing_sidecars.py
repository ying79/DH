#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
make_missing_sidecars.py — Create missing .meta.json sidecars for Aozora txt files.

Usage:
    python make_missing_sidecars.py --data-dir data
    python make_missing_sidecars.py --data-dir data --dry-run

What it does
------------
- Scan a folder (default: ./data) for *.txt files.
- For each txt, check if a sidecar 'xxx.meta.json' already exists:
    - if it exists → leave it untouched.
    - if it does NOT exist → create a new .meta.json file.
- It tries to *guess* title / author from the first non-empty lines
  of the txt. This is only a heuristic; you can edit the JSON later
  if needed.
- `card_url` is left empty ("") because we cannot infer it from the
  plain txt safely.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple, Optional


def guess_title_author(txt_path: Path, max_lines: int = 40) -> Tuple[str, str]:
    """
    Very simple heuristic:
    - Read the first `max_lines` lines.
    - Take the first non-empty line as `title`.
    - Take the next non-empty line as `author`.
    - If anything is missing, fall back to the file stem.
    """
    try:
        raw = txt_path.read_text(encoding="utf-8")
    except Exception:
        raw = txt_path.read_text(encoding="utf-8", errors="ignore")

    non_empty = []
    for line in raw.splitlines()[:max_lines]:
        s = line.strip()
        if s:
            non_empty.append(s)
        if len(non_empty) >= 2:
            break

    title = non_empty[0] if non_empty else txt_path.stem
    author = non_empty[1] if len(non_empty) >= 2 else ""

    return title, author


def build_sidecar_content(txt_path: Path) -> dict:
    """
    Build a minimal sidecar JSON object for a given txt file.
    `card_url` is left empty and can be filled manually later.
    """
    title, author = guess_title_author(txt_path)

    return {
        "title": title,
        "author": author,
        "card_url": "",
    }


def make_missing_sidecars(
    data_dir: Path,
    dry_run: bool = False,
) -> None:
    """
    Scan `data_dir` for *.txt files and create sidecars where missing.
    Existing sidecars are not modified.

    If `dry_run` is True, only print what would be done.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    txt_files = sorted(data_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under {data_dir}")
        return

    created = 0
    skipped = 0

    for txt in txt_files:
        sidecar = txt.with_suffix(".meta.json")
        if sidecar.exists():
            skipped += 1
            continue

        meta = build_sidecar_content(txt)

        if dry_run:
            print(f"[DRY-RUN] Would create: {sidecar}")
            print("          content:", json.dumps(meta, ensure_ascii=False))
        else:
            sidecar.write_text(
                json.dumps(meta, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            created += 1
            print(f"[OK] Created sidecar: {sidecar}")

    print("\nSummary")
    print("-------")
    print(f"Existing sidecars kept: {skipped}")
    print(f"New sidecars created : {created}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create missing .meta.json sidecars for Aozora txt files."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Folder containing *.txt files (default: ./data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files, only show what would be created.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    make_missing_sidecars(data_dir=data_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
