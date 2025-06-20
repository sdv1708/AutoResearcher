# tests/test_loader_processor.py
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from typing import Iterator, Dict, Any
from autoresearcher.data.loaders.pubmed_loader import PubMedLoader
from autoresearcher.data.processors.document_processor import DocumentProcessor

# --- CONFIG ---
xml_path = Path("src/autoresearcher/data/pubmed25n0001.xml")

# --- GCS MOCK (for local testing without actual bucket) ---
class MockBlob:
    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = path

    def download_as_text(self) -> str:
        return Path(self.path).read_text(encoding="utf-8")

class MockBucket:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def list_blobs(self, prefix: str = None) -> Iterator[MockBlob]:
        return [MockBlob("pubmed25n0001.xml", self.file_path)]

class MockStorageClient:
    def bucket(self, bucket_name: str) -> MockBucket:
        return MockBucket(xml_path)

# --- TEST PIPELINE ---
if __name__ == "__main__":
    loader = PubMedLoader(bucket_name="test-bucket", client=MockStorageClient())
    processor = DocumentProcessor()

    docs = loader.load_abstracts()
    docs_list = list(docs)

    print("Loaded Abstracts:")
    for doc in docs_list:
        print(doc)

    print("\nProcessed Docs:")
    for processed in map(processor.process_document, docs_list):
        print(processed)
