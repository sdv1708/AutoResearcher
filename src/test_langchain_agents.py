# src/test_langchain_agents.py
"""
Test script for LangChain-based agents with Mistral 7B
"""
import asyncio
import time
from datetime import datetime

from autoresearcher.agents.langchain_agent_manager import LangChainAgentManager
from autoresearcher.embeddings.faiss_index import FAISSIndexManager
from autoresearcher.embeddings.vertex_embeddings import VertexAIEmbeddings


async def setup_test_index():
    """Set up test data in FAISS index"""
    print("=== Setting up test data ===")

    embeddings = VertexAIEmbeddings()
    index = FAISSIndexManager()

    # Test documents about diabetes and COVID-19
    test_chunks = [
        {
            "chunk_id": "PMC001_chunk_0",
            "document_id": "PMC001",
            "text": "Type 2 diabetes mellitus affects millions worldwide. Metformin remains the first-line treatment, showing 70-80% efficacy in glycemic control. Recent studies highlight its cardiovascular benefits beyond glucose management.",
            "chunk_type": "abstract",
            "metadata": {
                "section": "abstract",
                "source": "pmc",
                "title": "Metformin Efficacy in T2DM: A 10-Year Review",
                "authors": ["Smith J", "Jones A"],
                "publication_date": "2023-01-15",
            },
            "source": "pmc",
        },
        {
            "chunk_id": "PMC002_chunk_0",
            "document_id": "PMC002",
            "text": "Comparative analysis of diabetes medications shows GLP-1 agonists achieve superior HbA1c reduction (1.5-2.0%) compared to metformin alone (1.0-1.5%). However, cost remains a significant barrier to adoption.",
            "chunk_type": "results",
            "metadata": {
                "section": "results",
                "source": "pmc",
                "title": "GLP-1 vs Metformin: Comparative Effectiveness",
                "authors": ["Johnson B"],
                "publication_date": "2023-06-20",
            },
            "source": "pmc",
        },
        {
            "chunk_id": "CORD19_001_chunk_0",
            "document_id": "CORD19_001",
            "text": "COVID-19 patients with diabetes face 2-3x higher mortality risk. Glycemic control during infection correlates with improved outcomes. Continuous glucose monitoring shows promise in ICU settings.",
            "chunk_type": "findings",
            "metadata": {
                "section": "findings",
                "source": "cord19",
                "title": "Diabetes and COVID-19 Outcomes",
                "authors": ["Lee C", "Park D"],
                "publication_date": "2023-03-10",
            },
            "source": "cord19",
        },
    ]

    # Generate embeddings and add to index
    texts = [chunk["text"] for chunk in test_chunks]
    embeddings_array = await embeddings.embed_texts(texts)
    index.add_embeddings(embeddings_array, test_chunks)
    index.save_index()

    print(f"Added {len(test_chunks)} test chunks to index")
    return embeddings, index


async def test_langchain_summarizer(manager: LangChainAgentManager):
    """Test LangChain summarizer agent"""
    print("\n=== Testing LangChain Summarizer Agent ===")

    # Test different summary types
    summary_types = ["findings", "clinical", "methodology", "comparative"]

    for summary_type in summary_types[:2]:  # Test first two
        print(f"\n--- Testing {summary_type} summary ---")

        response = await manager.process_request(
            "summarizer",
            {
                "query": "diabetes treatment metformin GLP-1",
                "summary_type": summary_type,
                "max_chunks": 10,
            },
        )

        if response.success:
            data = response.data
            print(f"✓ Success! Used model: {data.get('model_used', 'unknown')}")
            print(f"  Documents: {data['documents_referenced']}")
            print(f"  Chunks: {data['chunks_used']}")
            print(f"  Confidence: {data['confidence_score']}")
            print(f"\nSummary preview:\n{data['summary'][:300]}...")

            if data.get("key_points"):
                print("\nKey Points:")
                for point in data["key_points"][:3]:
                    print(f"  • {point}")
        else:
            print(f"✗ Error: {response.error}")


async def test_langchain_synthesizer(manager: LangChainAgentManager):
    """Test LangChain synthesizer agent"""
    print("\n\n=== Testing LangChain Synthesizer Agent ===")

    response = await manager.process_request(
        "synthesizer",
        {
            "query": "diabetes treatment effectiveness comparison",
            "analysis_types": ["consensus", "contradictions", "themes"],
            "max_documents": 10,
        },
    )

    if response.success:
        data = response.data
        print(f"✓ Success! Used model: {data.get('model_used', 'unknown')}")
        print(f"  Sources analyzed: {data['sources_count']}")
        print(f"  Confidence: {data['confidence_score']}")

        # Show analyses
        print("\nAnalyses performed:")
        for analysis_type, results in data["analyses"].items():
            print(f"\n{analysis_type.upper()}:")
            if analysis_type == "consensus":
                print(f"  Agreement score: {results.get('agreement_score', 0):.1%}")
                points = results.get("consensus_points", [])
                for point in points[:3]:
                    print(f"  • {point}")
            elif analysis_type == "contradictions":
                cont = results.get("contradictions", [])
                print(f"  Found {len(cont)} contradictions")
            elif analysis_type == "themes":
                themes = results.get("primary_themes", [])
                print(f"  Found {len(themes)} primary themes")

        print(f"\nSynthesis preview:\n{data['synthesis'][:400]}...")
    else:
        print(f"✗ Error: {response.error}")


async def test_pipeline(manager: LangChainAgentManager):
    """Test full pipeline with LangChain"""
    print("\n\n=== Testing Full LangChain Pipeline ===")

    pipeline = [
        {
            "agent": "summarizer",
            "params": {"summary_type": "comparative", "max_chunks": 15},
        },
        {"agent": "synthesizer", "params": {"analysis_types": ["consensus", "themes"]}},
    ]

    result = await manager.process_pipeline(
        query="diabetes metformin treatment outcomes", pipeline=pipeline
    )

    if result["success"]:
        print(f"✓ Pipeline completed successfully!")
        print(f"  Model used: {result['model_used']}")
        print(f"  Steps completed: {result['pipeline']}")

        # Show results from each step
        for step, step_result in result["results"].items():
            print(f"\n{step.upper()} Results:")
            if step == "summarizer":
                print(f"  Summary length: {len(step_result.get('summary', ''))}")
                print(f"  Key points: {len(step_result.get('key_points', []))}")
            elif step == "synthesizer":
                print(f"  Synthesis length: {len(step_result.get('synthesis', ''))}")
                print(
                    f"  Analyses completed: {list(step_result.get('analyses', {}).keys())}"
                )
    else:
        print(f"✗ Pipeline failed: {result['error']}")


async def test_model_switching(manager: LangChainAgentManager):
    """Test switching between different models"""
    print("\n\n=== Testing Model Switching ===")

    # Test with mock model first (fast)
    print("\n--- Testing with mock model ---")
    manager.switch_model("mock")

    response = await manager.process_request(
        "summarizer", {"query": "test query", "summary_type": "findings"}
    )
    print(f"Mock model result: {response.success}")

    # Switch back to Mistral
    print("\n--- Switching back to Mistral-7B ---")
    manager.switch_model("mistral-7b", use_8bit=True)
    print("Model switched successfully")


async def main():
    """Run all LangChain agent tests"""
    print("Starting LangChain Agent Tests with Mistral 7B\n")
    print("=" * 60)

    # Set up test data
    embeddings, index = await setup_test_index()

    # Create agent manager with Mistral 7B
    print("\n=== Initializing LangChain Agent Manager ===")
    print("Model: Mistral-7B-Instruct")
    print("Mode: 8-bit quantization")
    print(
        "Note: Using mock LLM for testing. Set model_name='mistral-7b' for real model"
    )

    manager = LangChainAgentManager(
        llm_model="mock",  # Change to "mistral-7b" for real model
        use_8bit=True,
        embeddings_service=embeddings,
        index_manager=index,
    )

    # Run tests
    await test_langchain_summarizer(manager)
    await test_langchain_synthesizer(manager)
    await test_pipeline(manager)
    await test_model_switching(manager)

    # Show statistics
    print("\n\n=== Agent Statistics ===")
    stats = manager.get_agent_stats()
    print(f"Total requests: {stats['total_requests']}")
    print(f"Success rate: {stats['success_rate']*100:.1f}%")
    print(f"Current model: {stats['model']}")

    print("\n✓ All tests completed!")


if __name__ == "__main__":
    # For real Mistral 7B usage:
    # 1. Install: pip install transformers torch accelerate bitsandbytes
    # 2. Change llm_model="mock" to llm_model="mistral-7b"
    # 3. Ensure you have GPU with ~8GB VRAM (with 8-bit quantization)

    asyncio.run(main())
