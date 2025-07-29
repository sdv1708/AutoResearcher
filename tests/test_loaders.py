from pathlib import Path

from autoresearcher.data.loaders.pmc_loader import PMCXMLLoader
from autoresearcher.data.loaders.cord19_loader import CORD19Loader
from autoresearcher.data.chunking import chunk_document


SAMPLE_DIR = Path(__file__).parent / "data"
PMC_FILE = SAMPLE_DIR / "sample_pmc.xml"
CORD_FILE = SAMPLE_DIR / "sample_cord19.json"

def test_pytest_works():
    assert True

def test_pmc_loader():
    doc = PMCXMLLoader(PMC_FILE).load()
    assert doc.id and doc.title
    assert doc.source == "pmc"
    chunks = chunk_document(doc, chunk_size=256, overlap=32)
    assert len(chunks) > 0


def test_cord_loader():
    doc = CORD19Loader(CORD_FILE).load()
    assert doc.id and doc.title
    assert doc.source == "cord19"
    chunks = chunk_document(doc, chunk_size=256, overlap=32)
    assert len(chunks) > 0