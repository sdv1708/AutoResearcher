# AutoResearcher Project Context

## Project Overview
AutoResearcher is a medical domain research assistant built on Google Cloud Platform using RAG (Retrieval-Augmented Generation) and multi-agent AI for biomedical literature analysis.

## Current Status
- **Branch**: `feature/langchain-agents`
- **Phase**: Development - implementing core components with mock implementations
- **Development Mode**: Local testing without GPU/GCP requirements

## Architecture Overview

### Core Components
1. **Data Layer**: Document loaders, chunking, and preprocessing
2. **Embedding Layer**: Vertex AI embeddings (mocked) and FAISS indexing
3. **Agent Layer**: LangChain-based summarizer and synthesizer agents
4. **API Layer**: FastAPI REST endpoints
5. **Configuration**: Settings management and GCP configuration

### Technology Stack
- **Language**: Python 3.12
- **Framework**: FastAPI + LangChain
- **Vector Store**: FAISS
- **Cloud**: GCP Vertex AI (production)
- **Dependency Management**: Poetry

## Implementation Status

### ✅ Completed Components
1. **Project Structure**: Basic setup with Poetry, FastAPI skeleton
2. **Configuration Management**: Settings and GCP config classes
3. **Data Loaders**: PMC and CORD-19 loaders (implemented)
4. **Document Chunking**: Smart chunking with metadata preservation
5. **Embedding Pipeline**: End-to-end document processing
6. **FAISS Index**: Vector similarity search with persistence
7. **Vertex AI Integration**: Mock implementation for embeddings
8. **Agent Framework**: Base agent class and manager
9. **LLM Manager**: Model management with mock implementations
10. **Agent Implementation**: Summarizer and synthesizer agents
11. **Dataset Builder**: Flexible dataset creation from multiple sources
12. **Training Configuration**: Comprehensive config classes for all training aspects
13. **Data Formatters**: Template-based formatting for different model types
14. **Training Pipeline**: Prefect-orchestrated end-to-end training workflow
15. **Evaluation Metrics**: Medical domain-specific evaluation suite
16. **Test Suite**: Comprehensive testing framework for all components

### 🔄 Issues Identified and Fixed
1. **✅ FIXED**: Pydantic v2 compatibility issue in settings.py
2. **✅ FIXED**: Duplicate code in embedding pipeline text preparation
3. **✅ FIXED**: Added pydantic-settings dependency
4. **✅ FIXED**: LangChain deprecation warnings (updated imports to langchain-community)
5. **✅ FIXED**: KeyError in summarizer agent prompt initialization (using partial_variables)
6. **✅ FIXED**: Memory deprecation warnings (replaced with ChatMessageHistory)
7. **🔄 PENDING**: Potential circular imports in agent system
8. **🔄 PENDING**: Missing comprehensive error handling
9. **🔄 PENDING**: Inconsistent data schemas across loaders
10. **🔄 PENDING**: Type mismatches in data flow

### 🚧 Next Steps
1. Fix critical compatibility and flow issues
2. Implement comprehensive error handling
3. Standardize data schemas
4. Add proper testing framework
5. Prepare for production deployment

## File Structure
```
src/autoresearcher/
├── api/
│   └── main.py                 # FastAPI application
├── config/
│   ├── settings.py             # App configuration (FIXED)
│   └── gcp_config.py          # GCP configuration
├── data/
│   ├── chunking.py            # Document chunking logic
│   └── loaders/
│       ├── base_loader.py     # Base loader interface
│       ├── cord19_loader.py   # CORD-19 dataset loader
│       └── pmc_loader.py      # PMC dataset loader
├── embeddings/
│   ├── embedding_pipeline.py  # End-to-end embedding pipeline
│   ├── faiss_index.py        # FAISS index manager
│   └── vertex_embeddings.py  # Vertex AI embeddings (mocked)
├── agents/
│   ├── base_agent.py          # Base agent class
│   ├── langchain_agent_manager.py    # Agent coordination
│   ├── langchain_summarizer_agent.py # Document summarization
│   └── langchain_synthesizer_agent.py # Multi-document synthesis
├── llm/
│   └── llm_manager.py         # LLM management and model switching
├── training/                   # NEW: Training pipeline components
│   ├── dataset_builder.py     # Dataset creation from multiple sources
│   ├── training_config.py     # Configuration classes for training
│   ├── data_formats.py        # Data formatting templates
│   └── evaluation.py          # Evaluation metrics suite
├── pipelines/                  # NEW: Orchestration workflows
│   └── training_pipeline.py   # Prefect-based training pipeline
├── utils/                      # Utility functions
└── test_training_pipeline.py  # Comprehensive test suite
```

## Data Structures

### Document Format
```python
{
    "id": "PMC123456",
    "title": "Study Title",
    "abstract": "Abstract text",
    "sections": [
        {
            "title": "Introduction",
            "paragraphs": ["paragraph 1", "paragraph 2"],
            "subsections": [...]
        }
    ],
    "source": "pmc",
    "has_full_text": true,
    "metadata": {
        "authors": ["Smith J", "Doe A"],
        "publication_date": "2023-01-01",
        "journal": "Nature Medicine",
        "keywords": ["diabetes", "treatment"]
    }
}
```

### Chunk Format
```python
{
    "chunk_id": "PMC123_chunk_0",
    "document_id": "PMC123",
    "text": "chunk text",
    "chunk_index": 0,
    "chunk_type": "abstract",
    "metadata": {"section": "abstract", "source": "pmc"},
    "char_start": 0,
    "char_end": 500
}
```

## Key Design Decisions
1. **Mock-First Development**: All external services have mock implementations
2. **TODO-Driven Production**: Clear markers for cloud deployment
3. **Modular Architecture**: Independent, testable components
4. **Type Safety**: Comprehensive type annotations
5. **Async-First**: All I/O operations are async

## Development Workflow
1. **Local Development**: Uses mock implementations, no external dependencies
2. **Testing**: Component-level testing with mock data
3. **Production**: Enable real services through configuration
4. **Deployment**: GCP Cloud Run with Vertex AI integration

## Production Readiness Checklist
- [x] Fix Pydantic v2 compatibility
- [x] Add comprehensive testing suite
- [x] Implement training pipeline orchestration
- [x] Create evaluation metrics framework
- [ ] Implement comprehensive error handling
- [ ] Add retry logic for external services
- [ ] Standardize data schemas
- [ ] Add proper logging and monitoring
- [ ] Implement health checks
- [ ] Add authentication and authorization
- [ ] Configure CI/CD pipeline
- [ ] Set up monitoring and alerting
- [ ] Deploy to production environment

## Day 5 Implementation Summary

### Training Pipeline Components
1. **Dataset Builder** (`src/autoresearcher/training/dataset_builder.py`):
   - Base classes for flexible dataset creation
   - PubMedQA and BioASQ dataset builders
   - Instruction dataset generation from document chunks
   - Dataset merging with deduplication
   - Comprehensive statistics and validation

2. **Training Configuration** (`src/autoresearcher/training/training_config.py`):
   - ModelConfig for different model types (Mistral, BioBERT, Llama2)
   - LoRAConfig with preset configurations for different tasks
   - TrainingConfig with hyperparameter management
   - DataConfig for dataset processing settings
   - FullPipelineConfig for complete workflow management

3. **Data Formatters** (`src/autoresearcher/training/data_formats.py`):
   - Template-based formatting for different model types
   - Alpaca, Mistral, QA, Summarization formatters
   - Proper label masking for instruction tuning
   - DataFormatManager for batch processing

4. **Training Pipeline** (`src/autoresearcher/pipelines/training_pipeline.py`):
   - Prefect-orchestrated workflows
   - Environment validation and setup
   - Dataset preparation, training, evaluation, deployment flows
   - Error handling and recovery mechanisms
   - Complete end-to-end pipeline management

5. **Evaluation Suite** (`src/autoresearcher/training/evaluation.py`):
   - Medical domain-specific metrics
   - Exact match, F1, ROUGE, BERTScore implementations
   - Medical relevance scoring
   - Comprehensive evaluation reports with confidence intervals

6. **Test Suite** (`src/test_training_pipeline.py`):
   - Comprehensive testing for all training components
   - Mock implementations for testing without external dependencies
   - Integration tests for complete workflows
   - Performance and validation testing

### Key Features
- **Mock-First Development**: All components work locally without GPU/GCP
- **Comprehensive Configuration**: Preset configurations for different scenarios
- **Robust Evaluation**: Medical domain-specific metrics and benchmarks
- **Orchestrated Workflows**: Prefect-based pipeline management
- **Production Ready**: Clear TODO markers for cloud deployment

## Notes
- All cloud-specific code is commented with TODO markers
- Error messages are designed to be helpful for debugging
- Following Python best practices (PEP 8, type hints)
- Modular design allows for easy component replacement
- Complete training pipeline ready for LoRA/QLoRA fine-tuning
- Evaluation metrics optimized for medical AI applications
