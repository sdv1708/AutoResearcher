"""
================================================================================
  File Name      : embedding_pipeline.py
  Created On     : 2025-06-21
  Created By     : Pranav Rajesh Charakondala
  Description    :
    This module defines an EmbeddingPipeline class that:
    - Loads biomedical documents using a loader (e.g. PubMedLoader)
    - Cleans and chunks documents using DocumentProcessor
    - Converts document chunks into embeddings using Hugging Face Transformers
    - Optionally saves them to disk or a FAISS index
================================================================================
"""

from typing import List, Dict, Any, Optional
from autoresearcher.data.loaders.pubmed_loader import PubMedLoader
from autoresearcher.data.processors.document_processor import DocumentProcessor
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
import torch
import json
import os
import logging
import argparse
import numpy as np

# Try to import faiss if available
try:
    import faiss
except ImportError:
    faiss = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingPipeline:
    def __init__(self, bucket_name: str, model_name: str = "bert-base-uncased", client: Optional[Any] = None):
        self.loader = PubMedLoader(bucket_name=bucket_name, client=client)
        self.processor = DocumentProcessor(tokenizer_name=model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def run(self, prefix: str = "baseline/") -> List[Dict[str, Any]]:
        results = []
        embeddings = []
        for raw_doc in tqdm(self.loader.load_abstracts(prefix=prefix), desc="Processing documents"):
            processed_doc = self.processor.process_document(raw_doc)

            if "covid" not in processed_doc.get("mesh_terms", []) and processed_doc.get("language") != "eng":
                continue  # Skip if not English or not relevant

            for chunk in processed_doc["chunks"]:
                embedding = self._embed_text(chunk)
                embeddings.append(embedding)
                results.append({
                    "doc_id": processed_doc["id"],
                    "chunk": chunk,
                    "embedding": embedding.detach().cpu().numpy().tolist()
                })
        return results, embeddings

    def _embed_text(self, text: str) -> torch.Tensor:
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token

    def save_embeddings(self, results: List[Dict[str, Any]], path: str = "embeddings.json"):
        with open(path, "w") as f:
            json.dump(results, f)
        logger.info(f"Embeddings saved to {path}")

    def build_faiss_index(self, embeddings: List[torch.Tensor]):
        if faiss is None:
            raise ImportError("FAISS is not installed. Please install it to use this method.")
        if not embeddings:
            raise ValueError("No embeddings provided for FAISS index creation.")
        vectors = torch.cat(embeddings).detach().cpu().numpy()
        dim = vectors.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(vectors)
        logger.info(f"FAISS index built with {len(vectors)} vectors of dimension {dim}")
        return index


def main():
    parser = argparse.ArgumentParser(description="Embedding pipeline for biomedical documents")
    parser.add_argument("--bucket", type=str, required=True, help="Bucket name")
    parser.add_argument("--prefix", type=str, default="baseline/", help="Path prefix inside bucket")
    parser.add_argument("--out_path", type=str, default="embeddings.json", help="Output file for embeddings")
    parser.add_argument("--save_faiss", action="store_true", help="Whether to save to FAISS")
    args = parser.parse_args()

    pipeline = EmbeddingPipeline(bucket_name=args.bucket)
    results, embeddings = pipeline.run(prefix=args.prefix)
    pipeline.save_embeddings(results, path=args.out_path)

    if args.save_faiss:
        if faiss is None:
            raise ImportError("FAISS is not installed. Cannot save FAISS index.")
        index = pipeline.build_faiss_index(embeddings)
        faiss.write_index(index, "faiss_index.idx")
        logger.info("Saved FAISS index to faiss_index.idx")


if __name__ == "__main__":
    main()
