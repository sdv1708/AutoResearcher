"""Pipeline for processing documents through embeddings to index"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from ..data.chunking import Chunk, DocumentChunker
from .faiss_index import FAISSIndexManager
from .vertex_embeddings import VertexAIEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    """
    Pipeline for processing documents through embeddings to index
    """

    def __init__(
        self,
        embedding_service: Optional[VertexAIEmbeddings] = None,
        index_manager: Optional[FAISSIndexManager] = None,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.embeddings = embedding_service or VertexAIEmbeddings()
        self.index = index_manager or FAISSIndexManager()
        self.chunker = DocumentChunker()

        self.stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "embeddings_generated": 0,
            "errors": 0,
        }

    async def process_documents(
        self,
        documents: List[Dict[str, Any]],
        batch_size: int = 10,
        save_interval: int = 100,
    ) -> Dict[str, Any]:
        """
        Process a list of documents through the embedding pipeline
        """
        logger.info(
            f"Starting document processing pipeline - {len(documents)} documents to process"
        )

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]

            try:
                await self._process_batch(batch)

                if (i + batch_size) % save_interval == 0:
                    logger.info(
                        f"Processed {i + batch_size} documents, saving index state"
                    )
                    self.index.save_index()

            except Exception as e:
                logger.error(f"Error processing batch {i // batch_size}: {e}")
                self.stats["errors"] += len(batch)

        self.index.save_index()
        logger.info("Document processing complete")
        return self.stats

    async def _process_batch(self, documents: List[Dict[str, Any]]):
        """Process a single batch of documents"""

        all_chunks = []
        all_texts = []

        # Chunk documents into manageable pieces
        for doc in documents:
            try:
                chunks = self.chunker.chunk_document(doc)

                for chunk in chunks:
                    embedding_text = self._prepare_embedding_text(chunk)
                    all_chunks.append(embedding_text)

                    # Store chunk metadata
                    chunk_dict = chunk.to_dict()
                    chunk_dict["source"] = doc["id"]
                    all_texts.append(chunk_dict)

                self.stats["chunks_created"] += len(chunks)
            except Exception as e:
                logger.error(f"Error chunking document {doc.get('id', 'unknown')}: {e}")
                self.stats["errors"] += 1
                continue

        # Generate embeddings for all chunks
        if all_texts:
            try:
                embeddings = await self.embeddings.embed_texts(
                    all_chunks, task_type="RETRIEVAL_DOCUMENT"
                )

                # add embeddings to index
                self.index.add_embeddings(embeddings, all_chunks)

                self.stats["embeddings_generated"] += len(embeddings)
                self.stats["documents_processed"] += len(documents)

            except Exception as e:
                logger.error(f"Error generating embeddings: {e}")
                self.stats["errors"] += len(all_texts)

    def _prepare_embedding_text(self, chunk: Chunk) -> str:
        """
        Prepare text for embedding, ensuring it fits within model constraints
        """
        parts = []

        # Add section context
        section = chunk.metadata.get("section", "")
        if section and section != "Unknown Section":
            parts.append(f"[Section: {section}]")

        # Add chunk type if not a regular paragraph
        if chunk.chunk_type in ["title", "abstract"]:
            parts.append(f"[Type: {chunk.chunk_type}]")

        # Add text content
        parts.append(chunk.text)

        return " ".join(parts).strip()

    async def search(
        self,
        query: str,
        filter_source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the index for relevant documents based on a query
        """
        logger.info(f"Searching index for query: {query}")

        # Generate embedding for the query
        query_embedding = await self.embeddings.embed_query(query)

        # One line filter function to filter results by source
        filter_fn = lambda doc: (
            doc["source"] == filter_source if filter_source else None
        )

        # Perform search in the index
        results = self.index.search(query_embedding, filter_fn=filter_fn)

        logger.info(f"Found {len(results)} results for query")
        return results
