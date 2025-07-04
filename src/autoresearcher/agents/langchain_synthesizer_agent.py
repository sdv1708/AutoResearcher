# src/autoresearcher/agents/langchain_synthesizer_agent.py
"""
Synthesizer Agent using LangChain for multi-document analysis
"""
import json
from typing import Any, Dict, List, Optional

from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

from .base_agent import BaseLangChainAgent
from .langchain_summarizer_agent import LangChainSummarizerAgent


class ConsensusAnalysis(BaseModel):
    """Schema for consensus analysis output"""

    consensus_points: List[str] = Field(
        description="Points of agreement across studies"
    )
    agreement_score: float = Field(description="Overall agreement score 0-1")
    supporting_documents: int = Field(description="Number of supporting documents")


class ContradictionAnalysis(BaseModel):
    """Schema for contradiction analysis output"""

    contradictions: List[Dict[str, Any]] = Field(
        description="List of contradictions found"
    )
    severity: str = Field(description="Overall severity: low, moderate, high")


class ThemeAnalysis(BaseModel):
    """Schema for theme analysis output"""

    primary_themes: List[Dict[str, Any]] = Field(description="Main themes identified")
    emerging_themes: List[str] = Field(description="Emerging or minor themes")


class LangChainSynthesizerAgent(BaseLangChainAgent):
    """
    Synthesizer agent using LangChain for complex multi-document analysis
    """

    def __init__(
        self, llm_manager, summarizer_agent: Optional[LangChainSummarizerAgent] = None
    ):
        super().__init__("langchain_synthesizer", llm_manager)

        self.summarizer = summarizer_agent or LangChainSummarizerAgent(llm_manager)

        # Initialize analysis chains
        self.analysis_chains = self._initialize_analysis_chains()

        # Initialize synthesis chain
        self.synthesis_chain = self._initialize_synthesis_chain()

    def _initialize_analysis_chains(self) -> Dict[str, LLMChain]:
        """Initialize chains for different analyses"""
        chains = {}

        # Consensus analysis chain
        consensus_prompt = PromptTemplate(
            input_variables=["sources", "query"],
            template="""Analyze these research papers about '{query}' and identify points of consensus.

Sources:
{sources}

Identify:
1. What findings do most papers agree on?
2. What methodologies are commonly used?
3. What conclusions are consistent across studies?

Provide the analysis in the following JSON format:
{{
    "consensus_points": ["point 1", "point 2", ...],
    "agreement_score": 0.0-1.0,
    "supporting_documents": number
}}
""",
        )

        chains["consensus"] = LLMChain(
            llm=self.llm, prompt=consensus_prompt, output_key="consensus_analysis"
        )

        # Contradiction analysis chain
        contradiction_prompt = PromptTemplate(
            input_variables=["sources", "query"],
            template="""Analyze these research papers about '{query}' and identify contradictions.

Sources:
{sources}

Identify conflicting findings, methodologies, or conclusions.

Provide the analysis in JSON format:
{{
    "contradictions": [
        {{
            "topic": "what the contradiction is about",
            "description": "detailed description",
            "sources_involved": ["source1", "source2"],
            "severity": "low/moderate/high"
        }}
    ],
    "severity": "overall severity"
}}
""",
        )

        chains["contradictions"] = LLMChain(
            llm=self.llm,
            prompt=contradiction_prompt,
            output_key="contradiction_analysis",
        )

        # Theme extraction chain
        theme_prompt = PromptTemplate(
            input_variables=["sources", "query"],
            template="""Extract major themes from these research papers about '{query}'.

Sources:
{sources}

Identify recurring topics, methodologies, and research directions.

Provide the analysis in JSON format:
{{
    "primary_themes": [
        {{
            "theme": "theme name",
            "description": "theme description",
            "frequency": 0.0-1.0,
            "example_sources": ["source1", "source2"]
        }}
    ],
    "emerging_themes": ["theme1", "theme2", ...]
}}
""",
        )

        chains["themes"] = LLMChain(
            llm=self.llm, prompt=theme_prompt, output_key="theme_analysis"
        )

        return chains

    def _initialize_synthesis_chain(self) -> LLMChain:
        """Initialize the final synthesis chain"""

        synthesis_prompt = PromptTemplate(
            input_variables=[
                "query",
                "consensus",
                "contradictions",
                "themes",
                "sources_count",
            ],
            template="""Create a comprehensive synthesis report for the research query: '{query}'

Based on analysis of {sources_count} sources:

Consensus Analysis:
{consensus}

Contradictions Found:
{contradictions}

Major Themes:
{themes}

Provide a detailed synthesis that:
1. Summarizes the current state of research
2. Highlights areas of agreement and disagreement
3. Identifies research gaps
4. Suggests future research directions
5. Provides clinical/practical implications

Synthesis Report:""",
        )

        return LLMChain(llm=self.llm, prompt=synthesis_prompt, output_key="synthesis")

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
        analysis_types = input_data.get("analysis_types", ["consensus", "themes"])
        max_documents = input_data.get("max_documents", 10)

        # Step 1: Get summarized content from summarizer
        self.logger.info(f"Getting summaries for synthesis: {query}")
        summary_response = await self.summarizer.process(
            {
                "query": query,
                "summary_type": "comparative",
                "max_chunks": max_documents * 5,
            }
        )

        if not summary_response.success:
            raise Exception(f"Summary generation failed: {summary_response.error}")

        summary_data = summary_response.data
        sources = summary_data.get("sources", [])

        if not sources:
            return {
                "synthesis": "No sources found to synthesize.",
                "analyses": {},
                "sources_count": 0,
            }

        # Step 2: Format sources for analysis
        formatted_sources = self._format_sources_for_analysis(sources, summary_data)

        # Step 3: Run requested analyses
        analyses = {}
        for analysis_type in analysis_types:
            if analysis_type in self.analysis_chains:
                self.logger.info(f"Running {analysis_type} analysis")
                analysis_result = await self._run_analysis(
                    analysis_type, formatted_sources, query
                )
                analyses[analysis_type] = analysis_result

        # Step 4: Generate synthesis
        synthesis = await self._generate_synthesis(query, analyses, len(sources))

        # Step 5: Create structured output
        structured_output = self._create_structured_output(sources, analyses)

        return {
            "synthesis": synthesis,
            "analyses": analyses,
            "structured_data": structured_output,
            "sources_count": len(sources),
            "summary_data": summary_data,
            "confidence_score": self._calculate_synthesis_confidence(sources, analyses),
            "model_used": self.llm_manager.model_name,
        }

    def _format_sources_for_analysis(
        self, sources: List[Dict[str, Any]], summary_data: Dict[str, Any]
    ) -> str:
        """Format sources for LLM analysis"""
        formatted_parts = []

        for i, source in enumerate(sources[:10], 1):  # Limit to 10 sources
            source_text = f"""Source {i}:
    ID: {source['document_id']}
    Title: {source.get('title', 'Unknown')}
    Authors: {', '.join(source.get('authors', ['Unknown']))}
    Date: {source.get('publication_date', 'Unknown')}
    Type: {source.get('source', 'Unknown')}
    Relevance: {source.get('relevance_score', 0):.2f}

    Key content: {self._get_source_content(source['document_id'], summary_data)}
    """
            formatted_parts.append(source_text)

        return "\n---\n".join(formatted_parts)

    def _get_source_content(self, doc_id: str, summary_data: Dict[str, Any]) -> str:
        """Extract key content for a source from summary data"""
        # In a real implementation, this would get the actual chunks
        # For now, return a placeholder
        key_points = summary_data.get("key_points", [])
        if key_points:
            return (
                key_points[0]
                if isinstance(key_points[0], str)
                else "Content summary available in full report"
            )
        return "Content summary available in full report"

    async def _run_analysis(
        self, analysis_type: str, formatted_sources: str, query: str
    ) -> Dict[str, Any]:
        """Run a specific analysis chain"""
        chain = self.analysis_chains[analysis_type]

        try:
            # Run the chain
            result = await chain.arun(sources=formatted_sources, query=query)

            # Parse the JSON output
            try:
                parsed_result = json.loads(result)
                return parsed_result
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract structured data
                self.logger.warning(
                    f"Failed to parse JSON for {analysis_type}, using fallback"
                )
                return self._fallback_parse(result, analysis_type)

        except Exception as e:
            self.logger.error(f"Error in {analysis_type} analysis: {e}")
            return self._get_default_analysis(analysis_type)

    def _fallback_parse(self, text: str, analysis_type: str) -> Dict[str, Any]:
        """Fallback parsing when JSON parsing fails"""
        if analysis_type == "consensus":
            return {
                "consensus_points": [
                    line.strip()
                    for line in text.split("\n")
                    if line.strip() and not line.startswith("{")
                ][:5],
                "agreement_score": 0.7,
                "supporting_documents": 5,
            }
        elif analysis_type == "contradictions":
            return {
                "contradictions": [
                    {
                        "topic": "Parsed from text",
                        "description": text[:200],
                        "sources_involved": ["multiple"],
                        "severity": "moderate",
                    }
                ],
                "severity": "moderate",
            }
        elif analysis_type == "themes":
            return {
                "primary_themes": [
                    {
                        "theme": "Extracted theme",
                        "description": "Theme extracted from analysis",
                        "frequency": 0.8,
                        "example_sources": ["multiple"],
                    }
                ],
                "emerging_themes": ["Additional themes found in analysis"],
            }
        return {}

    def _get_default_analysis(self, analysis_type: str) -> Dict[str, Any]:
        """Get default analysis when chain fails"""
        defaults = {
            "consensus": {
                "consensus_points": ["Analysis in progress"],
                "agreement_score": 0.5,
                "supporting_documents": 0,
            },
            "contradictions": {"contradictions": [], "severity": "unknown"},
            "themes": {"primary_themes": [], "emerging_themes": []},
        }
        return defaults.get(analysis_type, {})

    async def _generate_synthesis(
        self, query: str, analyses: Dict[str, Any], sources_count: int
    ) -> str:
        """Generate final synthesis using all analyses"""

        # Format analyses for synthesis
        consensus_text = self._format_analysis(
            analyses.get("consensus", {}), "consensus"
        )
        contradictions_text = self._format_analysis(
            analyses.get("contradictions", {}), "contradictions"
        )
        themes_text = self._format_analysis(analyses.get("themes", {}), "themes")

        # Run synthesis chain
        synthesis = await self.synthesis_chain.arun(
            query=query,
            consensus=consensus_text,
            contradictions=contradictions_text,
            themes=themes_text,
            sources_count=sources_count,
        )

        return synthesis

    def _format_analysis(self, analysis: Dict[str, Any], analysis_type: str) -> str:
        """Format analysis results for synthesis"""
        if analysis_type == "consensus":
            points = analysis.get("consensus_points", [])
            score = analysis.get("agreement_score", 0)
            return f"Agreement Score: {score:.1%}\nKey Points:\n" + "\n".join(
                f"- {p}" for p in points
            )

        elif analysis_type == "contradictions":
            contradictions = analysis.get("contradictions", [])
            if not contradictions:
                return "No significant contradictions found."

            formatted = []
            for c in contradictions[:3]:  # Limit to top 3
                formatted.append(
                    f"- {c.get('topic', 'Unknown')}: {c.get('description', '')[:100]}..."
                )
            return "\n".join(formatted)

        elif analysis_type == "themes":
            themes = analysis.get("primary_themes", [])
            if not themes:
                return "No major themes identified."

            formatted = []
            for t in themes[:3]:  # Limit to top 3
                formatted.append(
                    f"- {t.get('theme', 'Unknown')}: {t.get('description', '')[:100]}..."
                )
            return "\n".join(formatted)

        return "Analysis not available"

    def _create_structured_output(
        self, sources: List[Dict[str, Any]], analyses: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create structured data output"""
        return {
            "document_matrix": self._create_document_matrix(sources),
            "analysis_summary": {
                "consensus_strength": analyses.get("consensus", {}).get(
                    "agreement_score", 0
                ),
                "contradiction_count": len(
                    analyses.get("contradictions", {}).get("contradictions", [])
                ),
                "theme_count": len(
                    analyses.get("themes", {}).get("primary_themes", [])
                ),
            },
            "key_insights": self._extract_key_insights(analyses),
        }

    def _create_document_matrix(
        self, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Create document comparison matrix"""
        matrix = []
        for source in sources[:10]:
            matrix.append(
                {
                    "document_id": source["document_id"],
                    "year": (
                        source.get("publication_date", "Unknown")[:4]
                        if source.get("publication_date")
                        else "Unknown"
                    ),
                    "source_type": source.get("source", "Unknown"),
                    "relevance": round(source.get("relevance_score", 0), 3),
                }
            )
        return matrix

    def _extract_key_insights(self, analyses: Dict[str, Any]) -> List[str]:
        """Extract key insights from analyses"""
        insights = []

        # From consensus
        if "consensus" in analyses:
            score = analyses["consensus"].get("agreement_score", 0)
            if score > 0.8:
                insights.append("Strong consensus found across studies")
            elif score < 0.5:
                insights.append("Limited consensus - diverse findings")

        # From contradictions
        if "contradictions" in analyses:
            cont_count = len(analyses["contradictions"].get("contradictions", []))
            if cont_count > 3:
                insights.append(
                    f"Multiple contradictions ({cont_count}) suggest active debate"
                )

        # From themes
        if "themes" in analyses:
            theme_count = len(analyses["themes"].get("primary_themes", []))
            if theme_count > 5:
                insights.append(f"Rich research area with {theme_count} major themes")

        return insights

    def _calculate_synthesis_confidence(
        self, sources: List[Dict[str, Any]], analyses: Dict[str, Any]
    ) -> float:
        """Calculate confidence in synthesis results"""

        # Source quality factor
        source_factor = min(len(sources) / 10, 1.0)

        # Analysis completeness factor
        completed_analyses = sum(1 for a in analyses.values() if a)
        analysis_factor = completed_analyses / 3  # Assuming 3 main analyses

        # Agreement factor
        agreement_score = analyses.get("consensus", {}).get("agreement_score", 0.5)

        # Combined confidence
        confidence = 0.4 * source_factor + 0.3 * analysis_factor + 0.3 * agreement_score

        return round(confidence, 3)
