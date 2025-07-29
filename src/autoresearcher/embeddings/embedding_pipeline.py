"""
End-to-end helper that:
1. Accepts Document objects
2. Chunks → embeds → adds to FAISS
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np

from autoresearcher.core.models import Chunk, Document
from autoresearcher.data.chunking import chunk_document
from autoresearcher.embeddings.faiss_index import FAISSIndexManager
from autoresearcher.embeddings.vertex_embeddings import VertexEmbeddings


class EmbeddingPipeline:
    def __init__(self, dim: int = 768, index_type: str = "flat") -> None:
        self.embedder = VertexEmbeddings()
        self.index = FAISSIndexManager(dim, index_type)

    # ───────────────────────── ingest ──────────────────────────
    def ingest(self, docs: Iterable[Document]) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        for doc in docs:
            chunks = chunk_document(doc)
            all_chunks.extend(chunks)

        # batch embed
        texts = [c.text for c in all_chunks]
        vecs = self.embedder.embed(texts)
        metadata = [c.to_dict() for c in all_chunks]
        self.index.add(vecs.astype(np.float32), metadata)
        return all_chunks

    # ───────────────────────── search ──────────────────────────
    def search(self, query: str, k: int = 5):
        qvec = self.embedder.embed([query])
        return self.index.search(qvec, k)

    # ───────────────────────── persist ─────────────────────────
    def save(self, path: str | Path) -> None:
        self.index.save(path)

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingPipeline":
        self = cls()
        self.index = FAISSIndexManager.load(path)
        return self
