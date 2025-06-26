# src/autoresearcher/embeddings/faiss_index.py
"""
FAISS Index Manager for efficient similarity search
"""
import json
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

# TODO: Uncomment for GCS integration
# from google.cloud import storage

logger = logging.getLogger(__name__)


class FAISSIndexManager:
    """
    Manage FAISS indices for vector similarity search

    Supports:
    - Multiple index types (Flat, IVF, HNSW)
    - Persistence to local disk or GCS
    - Incremental updates
    - Metadata storage
    """

    def __init__(
        self,
        dimension: int = 768,
        index_type: str = "Flat",
        index_path: Optional[str] = None,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.index_path = Path(index_path) if index_path else Path("./data/faiss_index")
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Initialize index
        self.index = self._create_index(index_type)

        # Document metadata storage
        self.documents: List[Dict[str, Any]] = []
        self.id_to_idx: Dict[str, int] = {}  # Map document ID to index position

        # Index statistics
        self.stats = {
            "total_vectors": 0,
            "index_type": index_type,
            "dimension": dimension,
            "created_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
        }

        # TODO: For cloud deployment:
        # self.storage_client = storage.Client()
        # self.bucket_name = gcp_config.index_bucket

    def _create_index(self, index_type: str) -> faiss.Index:
        """Create FAISS index based on type"""

        if index_type == "Flat":
            # Exact search (no approximation)
            # Best for < 1M vectors
            index = faiss.IndexFlatL2(self.dimension)

        elif index_type == "IVF":
            # Inverted File Index
            # Good for 1M-10M vectors
            # TODO: Tune nlist based on dataset size
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)

        elif index_type == "HNSW":
            # Hierarchical Navigable Small World
            # Good for high recall, fast search
            index = faiss.IndexHNSWFlat(self.dimension, 32)  # 32 = M parameter

        elif index_type == "IVF_PQ":
            # Product Quantization for compression
            # Good for very large datasets (>10M)
            nlist = 100
            m = 8  # Number of subquantizers
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFPQ(quantizer, self.dimension, nlist, m, 8)

        else:
            raise ValueError(f"Unknown index type: {index_type}")

        logger.info(f"Created FAISS {index_type} index with dimension {self.dimension}")
        return index

    def add_embeddings(
        self, embeddings: np.ndarray, documents: List[Dict[str, Any]]
    ) -> int:
        """
        Add embeddings and documents to index

        Args:
            embeddings: Numpy array of shape (n_docs, dimension)
            documents: List of document metadata

        Returns:
            Number of vectors added
        """
        if len(embeddings) != len(documents):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and documents ({len(documents)}) count mismatch"
            )

        # Ensure float32
        embeddings = embeddings.astype(np.float32)

        # Train index if needed (for IVF indices)
        if hasattr(self.index, "is_trained") and not self.index.is_trained:
            logger.info("Training FAISS index...")
            self.index.train(embeddings)

        # Add to index
        start_idx = len(self.documents)
        self.index.add(embeddings)

        # Store metadata
        for i, doc in enumerate(documents):
            doc_copy = doc.copy()
            doc_copy["_faiss_idx"] = start_idx + i
            self.documents.append(doc_copy)

            # Map document ID to index
            if "id" in doc:
                self.id_to_idx[doc["id"]] = start_idx + i

        # Update stats
        self.stats["total_vectors"] = len(self.documents)
        self.stats["last_updated"] = datetime.utcnow().isoformat()

        logger.info(
            f"Added {len(embeddings)} vectors to index. Total: {self.stats['total_vectors']}"
        )
        return len(embeddings)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        filter_fn: Optional[callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents

        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            filter_fn: Optional function to filter results

        Returns:
            List of documents with similarity scores
        """
        # Ensure correct shape and type
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)

        # Search (get more results if filtering)
        search_k = k * 3 if filter_fn else k
        search_k = min(search_k, len(self.documents))

        if search_k == 0:
            return []

        # Perform search
        distances, indices = self.index.search(query_embedding, search_k)

        # Process results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid result
                continue

            doc = self.documents[idx].copy()
            doc["_score"] = float(dist)
            doc["_similarity"] = float(1 / (1 + dist))  # Convert distance to similarity

            # Apply filter if provided
            if filter_fn and not filter_fn(doc):
                continue

            results.append(doc)

            if len(results) >= k:
                break

        return results

    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID"""
        idx = self.id_to_idx.get(doc_id)
        if idx is not None and idx < len(self.documents):
            return self.documents[idx].copy()
        return None

    def save_index(self, path: Optional[str] = None):
        """
        Save index to disk

        TODO: For cloud deployment:
        - Save to GCS instead of local disk
        - Use versioning for index files
        - Implement atomic updates
        """
        save_path = Path(path) if path else self.index_path
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_file = save_path / "index.faiss"
        faiss.write_index(self.index, str(index_file))

        # Save metadata
        metadata = {
            "documents": self.documents,
            "id_to_idx": self.id_to_idx,
            "stats": self.stats,
        }
        metadata_file = save_path / "metadata.pkl"
        with open(metadata_file, "wb") as f:
            pickle.dump(metadata, f)

        # Save stats as JSON for easy inspection
        stats_file = save_path / "stats.json"
        with open(stats_file, "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.info(f"Saved index to {save_path}")

        # TODO: Cloud save
        # self._save_to_gcs(save_path)

    def load_index(self, path: Optional[str] = None):
        """Load index from disk"""
        load_path = Path(path) if path else self.index_path

        # Load FAISS index
        index_file = load_path / "index.faiss"
        if index_file.exists():
            self.index = faiss.read_index(str(index_file))
            logger.info(f"Loaded FAISS index from {index_file}")

        # Load metadata
        metadata_file = load_path / "metadata.pkl"
        if metadata_file.exists():
            with open(metadata_file, "rb") as f:
                metadata = pickle.load(f)
                self.documents = metadata["documents"]
                self.id_to_idx = metadata["id_to_idx"]
                self.stats = metadata["stats"]
            logger.info(f"Loaded {len(self.documents)} documents from metadata")

    def remove_documents(self, doc_ids: List[str]) -> int:
        """
        Remove documents from index

        Note: FAISS doesn't support direct removal, so we rebuild
        """
        # TODO: Implement more efficient removal for large indices
        # Could use IndexIDMap2 for direct removal support

        removed_count = 0
        remaining_docs = []
        remaining_embeddings = []

        # Collect remaining documents
        for i, doc in enumerate(self.documents):
            if doc.get("id") not in doc_ids:
                remaining_docs.append(doc)
                # TODO: Extract embedding from index
                # remaining_embeddings.append(...)
            else:
                removed_count += 1

        logger.info(f"Removed {removed_count} documents from index")

        # TODO: Rebuild index with remaining documents
        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            **self.stats,
            "current_vectors": len(self.documents),
            "index_size_bytes": self.index.ntotal * self.dimension * 4,  # float32
            "documents_by_source": self._count_by_source(),
        }

    def _count_by_source(self) -> Dict[str, int]:
        """Count documents by source"""
        counts = {}
        for doc in self.documents:
            source = doc.get("source", "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts
