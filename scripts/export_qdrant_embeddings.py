"""Export existing Qdrant SQLite embeddings to compact numpy format.

Reads the pickled PointStruct blobs directly from storage.sqlite — no QdrantClient,
no HNSW index, no RAM OOM. Run once from the repo root or scripts/ directory.

Output (scripts/embeddings_numpy/):
    vectors.npy   — float16 array, shape (N, max_patches, 320), zero-padded
    metadata.json — list of {path, doc_id} dicts, one per point
"""

import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "embeddings/collection/colqwen_multi_pages/storage.sqlite"
OUT_DIR = Path(__file__).parent / "embeddings_numpy"


def main() -> None:
    conn = sqlite3.connect(DB_PATH)

    n_points = conn.execute("SELECT COUNT(*) FROM points").fetchone()[0]
    print(f"Found {n_points} points in {DB_PATH}")

    # Peek at first point to get n_dims
    first_obj = pickle.loads(conn.execute("SELECT point FROM points LIMIT 1").fetchone()[0])
    n_dims = len(first_obj.vector[0])

    # Pass 1: find max patch count (pages have variable patch counts)
    print("Pass 1/2: scanning for max patch count...")
    max_patches = 0
    for (blob,) in conn.execute("SELECT point FROM points ORDER BY rowid"):
        max_patches = max(max_patches, len(pickle.loads(blob).vector))
    print(f"  n_dims={n_dims}, max_patches={max_patches}")
    print(f"Pre-allocating ({n_points}, {max_patches}, {n_dims}) float16 "
          f"≈ {n_points * max_patches * n_dims * 2 / 1e9:.2f} GB")

    # Pass 2: fill array, zero-padding shorter pages
    vectors = np.zeros((n_points, max_patches, n_dims), dtype=np.float16)
    metadata = []

    print("Pass 2/2: extracting vectors...")
    for i, (blob,) in enumerate(conn.execute("SELECT point FROM points ORDER BY rowid")):
        obj = pickle.loads(blob)
        v = np.array(obj.vector, dtype=np.float16)
        vectors[i, : v.shape[0], :] = v   # zero-pad if fewer patches than max
        metadata.append({
            "path": obj.payload.get("path", ""),
            "doc_id": obj.payload.get("doc_id", ""),
        })
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n_points}")

    conn.close()

    OUT_DIR.mkdir(exist_ok=True)
    np.save(OUT_DIR / "vectors.npy", vectors)
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f)

    print(f"\n✅ Exported to {OUT_DIR}/")
    print(f"   vectors.npy  : {vectors.shape}  ({vectors.nbytes / 1e9:.2f} GB)")
    print(f"   metadata.json: {len(metadata)} entries")


if __name__ == "__main__":
    main()
