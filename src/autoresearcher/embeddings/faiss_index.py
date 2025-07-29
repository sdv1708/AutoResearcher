"""
FAISSIndexManager
─────────────────
*   Wraps every base index with `faiss.IndexIDMap` so we can supply our own
    64-bit IDs via `add_with_ids()`.  
*   Supports Flat L2 (MVP), IVF_Flat, and HNSW.  
*   Stores metadata in an in-memory dict `{id: meta}`; can be swapped for
    DuckDB / AlloyDB later.

All cloud-specific persistence tweaks are marked with  # TODO(cloud)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import faiss
import numpy as np


class FAISSIndexManager:
    def __init__(self, dim: int, index_type: str = "flat") -> None:
        self.dim = dim
        self.index_type = index_type.lower()
        self._id_map: Dict[int, Dict] = {}
        self._next_id = 0

        # ── choose base index ──────────────────────────────────────────
        if self.index_type == "flat":
            base = faiss.IndexFlatL2(dim)

        elif self.index_type == "ivf":
            nlist = 256
            quantizer = faiss.IndexFlatL2(dim)
            base = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

        elif self.index_type == "hnsw":
            base = faiss.IndexHNSWFlat(dim, 32)

        else:
            raise ValueError(f"Unsupported index_type={index_type}")

        # Wrap with ID map so we control vector IDs
        self.index = faiss.IndexIDMap(base)

    # ───────────────────────── CRUD ────────────────────────────────
    def add(self, vectors: np.ndarray, metadata: Sequence[Dict]) -> None:
        """Add N vectors with caller-supplied metadata."""
        if len(vectors) != len(metadata):
            raise ValueError("vectors and metadata length mismatch")

        # Train IVF/HNSW on the first batch if needed
        if not self.index.is_trained:
            self.index.train(vectors.astype(np.float32))

        ids = np.arange(self._next_id, self._next_id + len(vectors))
        self.index.add_with_ids(vectors.astype(np.float32), ids.astype(np.int64))

        for i, meta in zip(ids, metadata):
            self._id_map[int(i)] = meta

        self._next_id += len(vectors)

    def search(self, query_vecs: np.ndarray, k: int = 5) -> List[List[Dict]]:
        """Return list[ list[metadata + score] ] for each query."""
        D, I = self.index.search(query_vecs.astype(np.float32), k)
        results: List[List[Dict]] = []

        for ids, dists in zip(I, D):
            hits = []
            for idx, dist in zip(ids, dists):
                if idx == -1:  # empty slot
                    continue
                meta = dict(self._id_map[idx])
                meta["score"] = float(dist)
                hits.append(meta)
            results.append(hits)

        return results

    # ───────────────────────── persistence ────────────────────────
    def save(self, path: str | Path) -> None:
        """Save index + side-car metadata; ready for GCS upload."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path))

        # store id→metadata as numpy object array
        np.save(
            path.with_suffix(".meta.npy"),
            np.array(list(self._id_map.items()), dtype=object),
        )

        # TODO(cloud): `gsutil cp {path}* gs://<bucket>/indexes/` after build

    @classmethod
    def load(cls, path: str | Path) -> "FAISSIndexManager":
        """Load from disk (or GCS-downloaded file)."""
        path = Path(path)
        index = faiss.read_index(str(path))

        # Create stub instance; type no longer important after load
        self = cls(index.d, "flat")
        self.index = index

        sidecar = path.with_suffix(".meta.npy")
        if sidecar.exists():
            data = np.load(sidecar, allow_pickle=True)
            self._id_map = {int(k): v for k, v in data}
            self._next_id = max(self._id_map) + 1

        return self