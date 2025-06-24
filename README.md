# AutoResearcher

A medical‑domain research assistant that uses Retrieval‑Augmented Generation (RAG) and multi‑agent AI to:

Retrieve relevant biomedical papers from PubMed, PMC, and CORD‑19

Summarize key findings with a fine‑tuned LLM (e.g., Mistral‑7B via LoRA)

Compare studies and propose next research steps

Built on GCP Vertex AI for embeddings, matching engine, and model training, and orchestrated with LangChain Agents. Data ingestion, chunking, vector indexing (FAISS), fine‑tuning, and deployment are automated for a seamless research workflow.

Quick Start

Clone this repo

git clone https://github.com/sdv1708/AutoResearcher.git
cd AutoResearcher

Configure config.yaml with your GCP project, storage paths, and model settings

More details on data ingestion, RAG setup, agent workflows, and fine‑tuning scripts are coming soon!
