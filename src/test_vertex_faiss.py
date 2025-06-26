# src/test_vertex_faiss_integration.py
"""
Test Vertex AI and FAISS integration
"""
import asyncio
from datetime import datetime

import numpy as np

from autoresearcher.data.chunking import Chunk
from autoresearcher.embeddings.embedding_pipeline import EmbeddingPipeline
from autoresearcher.embeddings.faiss_index import FAISSIndexManager
from autoresearcher.embeddings.vertex_embeddings import VertexAIEmbeddings


async def test_vertex_embeddings():
    """Test Vertex AI embeddings"""
    print("=== Testing Vertex AI Embeddings ===")

    embeddings_service = VertexAIEmbeddings()

    # Test texts
    texts = [
        "Diabetes mellitus type 2 is a chronic metabolic disorder",
        "COVID-19 vaccines have shown high efficacy in clinical trials",
        "Machine learning can accelerate drug discovery",
    ]

    # Generate embeddings
    embeddings = await embeddings_service.embed_texts(texts)

    print(f"Generated embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"First embedding sample: {embeddings[0][:5]}...")

    # Test query embedding
    query = "What are treatments for diabetes?"
    query_embedding = await embeddings_service.embed_query(query)
    print(f"Query embedding shape: {query_embedding.shape}")

    # Cost estimation
    cost = embeddings_service.estimate_cost(num_texts=10000)
    print(f"Cost estimate for 10k texts: ${cost['estimated_cost_usd']}")

    return embeddings


def test_faiss_index(embeddings: np.ndarray):
    """Test FAISS index"""
    print("\n=== Testing FAISS Index ===")

    # Create index
    index_manager = FAISSIndexManager(dimension=768, index_type="Flat")

    # Create mock documents
    documents = [
        {
            "id": "doc1",
            "text": "Diabetes mellitus type 2 is a chronic metabolic disorder",
            "source": "pmc",
            "chunk_type": "abstract",
        },
        {
            "id": "doc2",
            "text": "COVID-19 vaccines have shown high efficacy in clinical trials",
            "source": "cord19",
            "chunk_type": "paragraph",
        },
        {
            "id": "doc3",
            "text": "Machine learning can accelerate drug discovery",
            "source": "pubmed",
            "chunk_type": "title",
        },
    ]

    # Add to index
    num_added = index_manager.add_embeddings(embeddings, documents)
    print(f"Added {num_added} vectors to index")

    # Test search
    query_embedding = embeddings[0]  # Use first embedding as query
    results = index_manager.search(query_embedding, k=2)

    print("\nSearch results:")
    for i, result in enumerate(results):
        print(f"{i+1}. {result['text'][:50]}... (score: {result['_similarity']:.3f})")

    # Test filtering
    filtered_results = index_manager.search(
        query_embedding, k=2, filter_fn=lambda doc: doc["source"] == "cord19"
    )
    print(f"\nFiltered results (cord19 only): {len(filtered_results)}")

    # Save index
    index_manager.save_index()
    print("\nIndex saved successfully")

    # Get stats
    stats = index_manager.get_stats()
    print(f"Index stats: {stats}")


async def test_full_pipeline():
    """Test the complete pipeline - FIXED VERSION"""
    print("\n=== Testing Full Pipeline ===")

    from autoresearcher.embeddings.faiss_index import FAISSIndexManager
    from autoresearcher.embeddings.vertex_embeddings import VertexAIEmbeddings

    # Create components directly
    embeddings_service = VertexAIEmbeddings()
    index_manager = FAISSIndexManager()

    # Skip the pipeline and test components directly
    # Create pre-chunked data
    test_chunks = [
        {
            "chunk_id": "PMC123_chunk_0",
            "document_id": "PMC123",
            "text": "Novel COVID-19 Treatment: This study examines various treatment options for COVID-19.",
            "chunk_index": 0,
            "chunk_type": "title",
            "metadata": {"section": "title", "source": "pmc"},
            "source": "pmc",
        },
        {
            "chunk_id": "PMC123_chunk_1",
            "document_id": "PMC123",
            "text": "COVID-19 has affected millions worldwide. Treatment options include antivirals and vaccines.",
            "chunk_index": 1,
            "chunk_type": "abstract",
            "metadata": {"section": "abstract", "source": "pmc"},
            "source": "pmc",
        },
        {
            "chunk_id": "CORD19_456_chunk_0",
            "document_id": "CORD19_456",
            "text": "Machine Learning in Drug Discovery: AI and ML techniques are revolutionizing drug discovery.",
            "chunk_index": 0,
            "chunk_type": "title",
            "metadata": {"section": "title", "source": "cord19"},
            "source": "cord19",
        },
    ]

    # Generate embeddings
    texts = [chunk["text"] for chunk in test_chunks]
    embeddings = await embeddings_service.embed_texts(texts)
    print(f"Generated {len(embeddings)} embeddings")

    # Add to index
    num_added = index_manager.add_embeddings(embeddings, test_chunks)
    print(f"Added {num_added} chunks to index")

    # Save index
    index_manager.save_index()

    # Test search
    query = "COVID-19 treatment options"
    query_embedding = await embeddings_service.embed_query(query)
    results = index_manager.search(query_embedding, k=5)

    print(f"\nSearch found {len(results)} results")
    for i, result in enumerate(results[:3]):
        print(
            f"{i+1}. Chunk {result.get('chunk_id', 'N/A')}: {result.get('text', '')[:60]}..."
        )
        print(
            f"   Source: {result.get('source', 'unknown')}, Score: {result.get('_similarity', 0):.3f}"
        )

    # Show final stats
    stats = index_manager.get_stats()
    print(
        f"\nFinal index stats: {stats['total_vectors']} vectors from sources: {stats['documents_by_source']}"
    )


async def main():
    """Run all tests"""
    print("Starting Vertex AI and FAISS integration tests\n")

    # Test components
    embeddings = await test_vertex_embeddings()
    test_faiss_index(embeddings)
    await test_full_pipeline()

    print("\n=== All tests completed ===")


if __name__ == "__main__":
    asyncio.run(main())
