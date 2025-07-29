"""
Smoke-test the full embedding pipeline end-to-end.

▪  Loads a single PMC sample
▪  Ingests into pipeline (chunk → embed → index)
▪  Executes a search and checks we get at least one hit
"""

from pathlib import Path

from autoresearcher.data.loaders.pmc_loader import PMCXMLLoader
from autoresearcher.embeddings.embedding_pipeline import EmbeddingPipeline

SAMPLE_PMC = Path(__file__).parent / "data" / "sample_pmc.xml"


def test_pipeline_ingest_and_search() -> None:
    doc = PMCXMLLoader(SAMPLE_PMC).load()

    pipe = EmbeddingPipeline(index_type="flat")  # uses IndexIDMap internally
    pipe.ingest([doc])

    hits = pipe.search("cancer treatment", k=3)[0]

    assert len(hits) >= 1
    assert "chunk_id" in hits[0]  # metadata round-trip