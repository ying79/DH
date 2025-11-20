from pathlib import Path
import faiss, json

IDX = Path("index/faiss/vectors.faiss")
META = Path("index/faiss/metadata.jsonl")

idx = faiss.read_index(str(IDX))
print("Index type:", type(idx).__name__)
print("ntotal vectors:", idx.ntotal)
try:
    print("dimension:", idx.d)  # works for most index types
except Exception:
    pass
try:
    print("is_trained:", bool(idx.is_trained))
except Exception:
    pass
print("metadata lines:", sum(1 for _ in META.open("r", encoding="utf-8")))
