# src/autoresearcher/agents/langchain_summarizer_agent.py
"""
Summarizer Agent using LangChain
"""
from typing import Any, Dict, List, Optional

from langchain.chains import (LLMChain, MapReduceDocumentsChain,
                              StuffDocumentsChain)
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

from ..embeddings.faiss_index import FAISSIndexManager
from ..embeddings.vertex_embeddings import VertexAIEmbeddings
from .base_agent import BaseLangChainAgent


class LangChainSummarizerAgent(BaseLangChainAgent):
    """
    Summarizer agent using LangChain chains
    """

    def __init__(
        self,
        llm_manager,
        embeddings_service: Optional[VertexAIEmbeddings] = None,
        index_manager: Optional[FAISSIndexManager] = None,
    ):
        super().__init__("langchain_summarizer", llm_manager)

        self.embeddings = embeddings_service or VertexAIEmbeddings()
        self.index = index_manager or FAISSIndexManager()

        # Initialize prompts for different summary types
        self.prompts = self._initialize_prompts()

        # Initialize chains
        self.chains = self._initialize_chains()

    def _initialize_prompts(self) -> Dict[str, PromptTemplate]:
        """Initialize summary prompts"""
        # Get base template from LLM manager
        base_template = self.llm_manager.get_prompt_template("summary")

        prompts = {
            "findings": PromptTemplate(
                input_variables=["context", "question", "instruction", "output_format"],
                template=base_template.format(
                    instruction="Summarize the key findings from these medical research papers",
                    output_format="""Provide a comprehensive summary that includes:
1. Main findings across all papers
2. Any consensus or disagreements
3. Clinical implications
4. Gaps in current research

Summary:""",
                ),
            ),
            "clinical": PromptTemplate(
                input_variables=["context", "question", "instruction", "output_format"],
                template=base_template.format(
                    instruction="Provide a clinical summary of these research papers",
                    output_format="""Focus on:
1. Patient populations studied
2. Interventions and dosages
3. Clinical outcomes and efficacy
4. Safety profiles and adverse events
5. Recommendations for clinical practice

Clinical Summary:""",
                ),
            ),
            "methodology": PromptTemplate(
                input_variables=["context", "question", "instruction", "output_format"],
                template=base_template.format(
                    instruction="Analyze the research methodologies used in these papers",
                    output_format="""Include:
1. Study designs used
2. Sample sizes and populations
3. Statistical methods
4. Strengths and limitations
5. Methodological quality assessment

Methodology Analysis:""",
                ),
            ),
            "comparative": PromptTemplate(
                input_variables=["context", "question", "instruction", "output_format"],
                template=base_template.format(
                    instruction="Compare and contrast these research studies",
                    output_format="""Provide:
1. Similarities in approaches and findings
2. Key differences and contradictions
3. Relative strengths of each study
4. Overall synthesis of evidence

Comparative Summary:""",
                ),
            ),
        }

        return prompts

    def _initialize_chains(self) -> Dict[str, Any]:
        """Initialize LangChain chains for different operations"""
        chains = {}

        # Basic summarization chain
        chains["basic"] = load_summarize_chain(
            llm=self.llm, chain_type="stuff", verbose=True  # For small documents
        )

        # Map-reduce chain for longer documents
        chains["map_reduce"] = load_summarize_chain(
            llm=self.llm, chain_type="map_reduce", verbose=True
        )

        # Custom chains for each summary type
        for summary_type, prompt in self.prompts.items():
            chains[summary_type] = LLMChain(llm=self.llm, prompt=prompt, verbose=True)

        return chains

    def validate_input(self, input_data: Dict[str, Any]):
        """Validate input data"""
        required_fields = ["query"]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

    async def _process_with_langchain(
        self, input_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process using LangChain"""
        query = input_data["query"]
        summary_type = input_data.get("summary_type", "findings")
        max_chunks = input_data.get("max_chunks", 20)
        filters = input_data.get("filters", {})

        # Step 1: Retrieve relevant chunks
        self.logger.info(f"Retrieving chunks for query: {query}")
        relevant_chunks = await self._retrieve_relevant_chunks(
            query, max_chunks, filters
        )

        if not relevant_chunks:
            return {
                "summary": "No relevant documents found for your query.",
                "chunks_used": 0,
                "sources": [],
            }

        # Step 2: Convert to LangChain documents
        documents = self._chunks_to_documents(relevant_chunks)

        # Step 3: Generate summary using appropriate chain
        if len(documents) <= 3:
            # Use stuff chain for small number of docs
            summary = await self._generate_stuff_summary(documents, query, summary_type)
        else:
            # Use map-reduce for larger number
            summary = await self._generate_mapreduce_summary(
                documents, query, summary_type
            )

        # Step 4: Extract additional insights
        key_points = await self._extract_key_points_with_llm(documents, query)

        # Step 5: Get sources
        sources = self._extract_sources(relevant_chunks)

        return {
            "summary": summary,
            "key_points": key_points,
            "chunks_used": len(relevant_chunks),
            "documents_referenced": len(set(c["document_id"] for c in relevant_chunks)),
            "sources": sources,
            "summary_type": summary_type,
            "confidence_score": self._calculate_confidence(relevant_chunks),
            "model_used": self.llm_manager.model_name,
        }

    async def _retrieve_relevant_chunks(
        self, query: str, max_chunks: int, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks using embeddings"""
        # Generate query embedding
        query_embedding = await self.embeddings.embed_query(query)

        # Create filter function
        filter_fn = None
        if filters:

            def filter_fn(doc):
                if "sources" in filters and doc.get("source") not in filters["sources"]:
                    return False
                if "date_after" in filters:
                    pub_date = doc.get("metadata", {}).get("publication_date")
                    if pub_date and pub_date < filters["date_after"]:
                        return False
                return True

        # Search index
        chunks = self.index.search(query_embedding, k=max_chunks, filter_fn=filter_fn)

        return chunks

    def _chunks_to_documents(self, chunks: List[Dict[str, Any]]) -> List[Document]:
        """Convert chunks to LangChain documents"""
        documents = []

        for chunk in chunks:
            # Include metadata in page content for context
            metadata = chunk.get("metadata", {})
            section = metadata.get("section", "Unknown")

            # Format content with context
            content = f"[Section: {section}]\n{chunk['text']}"

            doc = Document(
                page_content=content,
                metadata={
                    "chunk_id": chunk.get("chunk_id"),
                    "document_id": chunk.get("document_id"),
                    "source": chunk.get("source"),
                    "score": chunk.get("_similarity", 0),
                    **metadata,
                },
            )
            documents.append(doc)

        return documents

    async def _generate_stuff_summary(
        self, documents: List[Document], query: str, summary_type: str
    ) -> str:
        """Generate summary using stuff chain (all docs at once)"""

        # Use custom chain for specific summary type
        if summary_type in self.chains:
            chain = self.chains[summary_type]

            # Combine documents
            context = "\n\n".join([doc.page_content for doc in documents])

            # Run chain
            result = await chain.arun(
                context=context,
                question=query,
                instruction="",  # Already in template
                output_format="",  # Already in template
            )

            return result
        else:
            # Use basic summarization
            chain = self.chains["basic"]
            result = await chain.arun(input_documents=documents)
            return result

    async def _generate_mapreduce_summary(
        self, documents: List[Document], query: str, summary_type: str
    ) -> str:
        """Generate summary using map-reduce chain"""

        # Configure map-reduce for medical summaries
        map_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Summarize the key medical findings from this section:

{text}

Key findings:""",
        )

        combine_prompt = self.prompts.get(summary_type, self.prompts["findings"])

        # Create custom map-reduce chain
        chain = MapReduceDocumentsChain(
            llm_chain=LLMChain(llm=self.llm, prompt=map_prompt),
            combine_document_chain=StuffDocumentsChain(
                llm_chain=LLMChain(llm=self.llm, prompt=combine_prompt),
                document_variable_name="context",
            ),
            document_variable_name="text",
            return_intermediate_steps=True,
        )

        # Run chain
        result = await chain.arun(input_documents=documents, question=query)

        return result

    async def _extract_key_points_with_llm(
        self, documents: List[Document], query: str
    ) -> List[str]:
        """Extract key points using LLM"""

        # Create key points extraction chain
        key_points_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on these research papers about {question}, extract 3-5 key points.

Context:
{context}

List the key points as bullet points:
""",
        )

        chain = LLMChain(llm=self.llm, prompt=key_points_prompt)

        # Get top documents
        top_docs = documents[:5]
        context = "\n\n".join([doc.page_content for doc in top_docs])

        # Extract key points
        result = await chain.arun(context=context, question=query)

        # Parse bullet points
        key_points = []
        for line in result.split("\n"):
            line = line.strip()
            if line and (
                line.startswith("-") or line.startswith("•") or line.startswith("*")
            ):
                key_points.append(line.lstrip("-•* "))

        return key_points[:5]  # Limit to 5

    def _extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique sources"""
        sources = {}

        for chunk in chunks:
            doc_id = chunk.get("document_id")
            if doc_id and doc_id not in sources:
                sources[doc_id] = {
                    "document_id": doc_id,
                    "source": chunk.get("source", "unknown"),
                    "title": chunk.get("metadata", {}).get("title", "Untitled"),
                    "authors": chunk.get("metadata", {}).get("authors", []),
                    "publication_date": chunk.get("metadata", {}).get(
                        "publication_date"
                    ),
                    "relevance_score": chunk.get("_similarity", 0),
                }

        return sorted(
            sources.values(), key=lambda x: x["relevance_score"], reverse=True
        )

    def _calculate_confidence(self, chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score"""
        if not chunks:
            return 0.0

        avg_similarity = sum(c.get("_similarity", 0) for c in chunks) / len(chunks)
        unique_docs = len(set(c.get("document_id") for c in chunks))
        doc_factor = min(unique_docs / 5, 1.0)

        confidence = 0.7 * avg_similarity + 0.3 * doc_factor
        return round(confidence, 3)
