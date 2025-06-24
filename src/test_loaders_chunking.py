# src/test_loaders_and_chunking.py
"""Test script for PMC and CORD-19 loaders with chunking"""
import asyncio
from pathlib import Path

from autoresearcher.data.chunking import DocumentChunker
from autoresearcher.data.loaders.cord19_loader import CORD19Loader
from autoresearcher.data.loaders.pmc_loader import PMCLoader


def test_pmc_loader():
    """Test PMC loader"""
    print("=== Testing PMC Loader ===")

    # Create sample PMC XML for testing
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <article>
        <front>
            <article-meta>
                <article-id pub-id-type="pmc">7654321</article-id>
                <title-group>
                    <article-title>Novel Approaches in Cancer Treatment</article-title>
                </title-group>
                <abstract>
                    <sec>
                        <title>Background</title>
                        <p>Cancer treatment has evolved significantly over the past decade.</p>
                   </sec>
                   <sec>
                       <title>Methods</title>
                       <p>We conducted a comprehensive review of recent therapeutic approaches.</p>
                   </sec>
               </abstract>
               <contrib-group>
                   <contrib contrib-type="author">
                       <name>
                           <surname>Johnson</surname>
                           <given-names>Emily R</given-names>
                       </name>
                   </contrib>
               </contrib-group>
               <pub-date date-type="pmc-release">
                   <year>2023</year>
                   <month>6</month>
                   <day>15</day>
               </pub-date>
               <journal-title>Cancer Research Journal</journal-title>
               <kwd-group>
                   <kwd>cancer</kwd>
                   <kwd>immunotherapy</kwd>
                   <kwd>targeted therapy</kwd>
               </kwd-group>
           </article-meta>
       </front>
       <body>
           <sec>
               <title>Introduction</title>
               <p>Cancer remains one of the leading causes of mortality worldwide. Recent advances in molecular biology have led to new therapeutic approaches.</p>
               <p>This review examines the latest developments in cancer treatment, focusing on immunotherapy and targeted molecular therapies.</p>
           </sec>
           <sec>
               <title>Current Treatment Modalities</title>
               <p>Traditional cancer treatments include surgery, chemotherapy, and radiation therapy. However, these approaches often have significant side effects.</p>
               <sec>
                   <title>Immunotherapy</title>
                   <p>Immunotherapy harnesses the body's immune system to fight cancer. Checkpoint inhibitors have shown remarkable success in treating various cancer types.</p>
               </sec>
           </sec>
       </body>
   </article>
   """

    # Save sample XML
    test_dir = Path("./test_data/pmc")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_file = test_dir / "PMC7654321.xml"
    with open(test_file, "w") as f:
        f.write(sample_xml)

    # Test loader
    loader = PMCLoader(data_dir=str(test_dir))

    for article in loader.load_articles(limit=1):
        print(f"\nLoaded PMC article: {article['id']}")
        print(f"Title: {article['title']}")
        print(f"Abstract sections: {len(article['abstract_sections'])}")
        print(f"Body sections: {len(article['sections'])}")
        print(f"Authors: {article['metadata']['authors']}")
        print(f"Keywords: {article['metadata'].get('keywords', [])}")

        return article  # Return for chunking test

    return None


def test_cord19_loader():
    """Test CORD-19 loader"""
    print("\n=== Testing CORD-19 Loader ===")

    # Create sample CORD-19 JSON
    sample_json = {
        "paper_id": "abc123def456",
        "metadata": {
            "title": "Understanding COVID-19 Transmission Dynamics",
            "authors": [
                {"first": "John", "middle": ["M"], "last": "Smith"},
                {"first": "Sarah", "middle": [], "last": "Johnson"},
            ],
            "doi": "10.1234/covid.2023.001",
            "pmcid": "PMC7890123",
            "pubmed_id": "33445678",
        },
        "abstract": [
            {
                "section": "Background",
                "text": "COVID-19 has caused a global pandemic affecting millions worldwide.",
            },
            {
                "section": "Objective",
                "text": "To understand the transmission dynamics of SARS-CoV-2 in different settings.",
            },
        ],
        "body_text": [
            {
                "section": "Introduction",
                "text": "The COVID-19 pandemic, caused by SARS-CoV-2, emerged in late 2019 and rapidly spread globally.",
            },
            {
                "section": "Introduction",
                "text": "Understanding transmission dynamics is crucial for developing effective control measures.",
            },
            {
                "section": "Methods",
                "text": "We analyzed transmission data from multiple countries using mathematical modeling approaches.",
            },
            {
                "section": "Results",
                "text": "Our analysis revealed significant variations in transmission rates across different settings.",
            },
        ],
        "bib_entries": {
            "ref1": {
                "title": "Initial genomic characterization of SARS-CoV-2",
                "authors": [{"first": "F", "last": "Wu"}],
                "venue": "Nature",
                "year": 2020,
            }
        },
    }

    # Save sample JSON
    test_dir = Path("./test_data/cord19/document_parses")
    test_dir.mkdir(parents=True, exist_ok=True)

    test_file = test_dir / "abc123def456.json"
    import json

    with open(test_file, "w") as f:
        json.dump(sample_json, f, indent=2)

    # Test loader
    loader = CORD19Loader(data_dir="./test_data/cord19")

    for paper in loader.load_papers(limit=1):
        print(f"\nLoaded CORD-19 paper: {paper['id']}")
        print(f"Title: {paper['title']}")
        print(f"Abstract length: {len(paper['abstract'])}")
        print(f"Number of sections: {len(paper['sections'])}")
        print(f"Authors: {paper['metadata']['authors']}")
        print(f"References: {len(paper['references'])}")

        return paper  # Return for chunking test

    return None


def test_chunking(documents):
    """Test document chunking"""
    print("\n=== Testing Document Chunking ===")

    chunker = DocumentChunker(
        chunk_size=300, chunk_overlap=50, min_chunk_size=50  # Smaller for testing
    )

    for doc in documents:
        if not doc:
            continue

        print(f"\nChunking document: {doc['id']}")
        chunks = chunker.chunk_document(doc)

        print(f"Total chunks created: {len(chunks)}")

        # Show chunk distribution
        chunk_types = {}
        for chunk in chunks:
            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1

        print("Chunk distribution:")
        for chunk_type, count in chunk_types.items():
            print(f"  {chunk_type}: {count}")

        # Show first few chunks
        print("\nFirst 3 chunks:")
        for chunk in chunks[:3]:
            print(f"\n  Chunk {chunk.chunk_index} ({chunk.chunk_type}):")
            print(f"    Text preview: {chunk.text[:100]}...")
            print(f"    Character range: {chunk.char_start}-{chunk.char_end}")
            print(f"    Section: {chunk.metadata.get('section', 'N/A')}")


def main():
    """Run all tests"""
    # Test PMC loader
    pmc_doc = test_pmc_loader()

    # Test CORD-19 loader
    cord19_doc = test_cord19_loader()

    # Test chunking on both documents
    documents = [pmc_doc, cord19_doc]
    test_chunking(documents)

    print("\n=== All tests completed ===")

    # Cleanup test data
    import shutil

    if Path("./test_data").exists():
        shutil.rmtree("./test_data")
        print("Test data cleaned up")


if __name__ == "__main__":
    main()
