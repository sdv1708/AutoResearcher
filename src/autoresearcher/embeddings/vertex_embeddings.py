"""
VertexEmbeddings
────────────────
A thin wrapper that works in two modes:

1. Local/dev  → returns deterministic pseudo-random vectors (no GCP cred needed)
2. Cloud/GCP  → calls Vertex AI    textembedding-gecko-@003

Switch with Settings.use_vertex (see core/settings.py).
"""

from __future__ import annotations

import hashlib
import os
from typing import Iterable, List

import numpy as np

from autoresearcher.core.settings import get_settings

try:
    # Light import so local dev doesn’t break if libs missing
    from vertexai.preview.language_models import TextEmbeddingModel  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    TextEmbeddingModel = None  # noqa: N816

settings = get_settings()


class VertexEmbeddings:
    """Callable embeddings client with batch support."""

    DIM = 768
    MAX_BATCH = 5

    def __init__(self) -> None:
        self._client = None
        if settings.use_vertex:
            self._bootstrap_vertex()

    # ─────────────────────── public api ────────────────────────
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return Nx768 float32 array (batched internally)."""
        if not texts:
            return np.empty((0, self.DIM), dtype=np.float32)

        if not settings.use_vertex:
            return self._mock_vectors(texts)

        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.MAX_BATCH):
            batch = texts[i : i + self.MAX_BATCH]
            resp = self._client.get_embeddings(batch, task_type="RETRIEVAL_DOCUMENT")
            embeddings.extend(np.asarray(e.values, dtype=np.float32) for e in resp)
        return np.vstack(embeddings)

    # ─────────────────────── private helpers ───────────────────
    def _bootstrap_vertex(self) -> None:
        # TODO(cloud): ensure VertexAPI enabled & service-account has role aiplatform.user
        if TextEmbeddingModel is None:
            raise RuntimeError("google-cloud-aiplatform not installed")
        project = settings.vertex_project_id
        region = settings.vertex_region
        if not (project and region):
            raise RuntimeError("Set VERTEX_PROJECT_ID & VERTEX_REGION env vars")
        os.environ["VERTEX_AI_PROJECT"] = project  # only for local override
        os.environ["VERTEX_AI_LOCATION"] = region
        self._client = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

    def _mock_vectors(self, texts: Iterable[str]) -> np.ndarray:
        """Deterministic mock so tests repeat."""
        vecs = []
        for t in texts:
            h = hashlib.sha256(t.encode()).digest()
            # take first DIM floats from hash bytes
            # seed = np.frombuffer(h, dtype=np.uint8)[: self.DIM].astype(np.float32)  # ORIGINAL: fails with DIM=768 > 32 bytes
            # Use hash as seed for deterministic random vectors
            rng = np.random.RandomState(np.frombuffer(h[:4], dtype=np.uint32)[0])
            vec = rng.rand(self.DIM).astype(np.float32)
            vecs.append(vec)
            # vecs.append(seed / 255.0)  # ORIGINAL
        return np.vstack(vecs)