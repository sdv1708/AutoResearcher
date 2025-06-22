# tests/test_embedding_pipeline.py
"""
================================================================================
  File Name      : test_embedding_pipeline.py
  Created On     : 2025-06-22
  Created By     : Pranav Rajesh Charakondala
  Description    :
    Test script to validate the full embedding pipeline using a mock GCS client.
    It loads a local XML file, processes documents, generates embeddings,
    and prints summary output.
================================================================================
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from autoresearcher.data.embedders.embedding_pipeline import EmbeddingPipeline
from autoresearcher.data.loaders.pubmed_loader import PubMedLoader
from autoresearcher.data.processors.document_processor import DocumentProcessor

# --- Mock GCS setup ---
class MockBlob:
    def __init__(self, name, path):
        self.name = name
        self.path = path

    def download_as_text(self):
        return Path(self.path).read_text(encoding="utf-8")

class MockBucket:
    def __init__(self, file_path):
        self.file_path = file_path

    def list_blobs(self, prefix=None):
        return [MockBlob("pubmed-2.xml", self.file_path)]

class MockStorageClient:
    def bucket(self, bucket_name):
        return MockBucket(str(Path(__file__).resolve().parents[1] / "src" / "autoresearcher" / "data" / "pubmed-2.xml"))


# --- Run test ---
if __name__ == "__main__":
    print("🔍 Running EmbeddingPipeline with mock data...")

    # Pass mock client directly
    pipeline = EmbeddingPipeline(
        bucket_name="dummy",
        model_name="bert-base-uncased",
        client=MockStorageClient()
    )

    results, _ = pipeline.run()

    print("\n Embedded Document Results:")
    for r in results:
        chunk_preview = r['chunk'][:75].replace("\n", " ")
        print(f" Doc ID: {r['doc_id']}, Chunk Preview: \"{chunk_preview}...\", Embedding Dim: {len(r['embedding'][0])}")
