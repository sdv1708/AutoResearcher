# src/autoresearcher/agents/langchain_agent_manager.py
"""
Agent Manager for LangChain-based agents
"""
import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationSummaryMemory

from ..embeddings.faiss_index import FAISSIndexManager
from ..embeddings.vertex_embeddings import VertexAIEmbeddings
from ..llm.llm_manager import LLMManager
from .base_agent import AgentResponse
from .langchain_summarizer_agent import LangChainSummarizerAgent
from .langchain_synthesizer_agent import LangChainSynthesizerAgent


class LangChainAgentManager:
    """
    Manages LangChain-based agents with shared resources
    """

    def __init__(
        self,
        llm_model: str = "mistral-7b",
        use_8bit: bool = True,
        embeddings_service: Optional[VertexAIEmbeddings] = None,
        index_manager: Optional[FAISSIndexManager] = None,
    ):
        # Initialize LLM manager
        self.llm_manager = LLMManager(model_name=llm_model, use_8bit=use_8bit)

        # Shared services
        self.embeddings = embeddings_service or VertexAIEmbeddings()
        self.index = index_manager or FAISSIndexManager()

        # Initialize agents
        self.agents = self._initialize_agents()

        # Conversation memory (shared across agents)
        self.conversation_memory = ConversationSummaryMemory(
            llm=self.llm_manager.get_langchain_llm(),
            memory_key="chat_history",
            return_messages=True,
        )

        # Request tracking
        self.request_history = []

    def _initialize_agents(self) -> Dict[str, Any]:
        """Initialize all agents"""
        agents = {}

        # Summarizer agent
        agents["summarizer"] = LangChainSummarizerAgent(
            llm_manager=self.llm_manager,
            embeddings_service=self.embeddings,
            index_manager=self.index,
        )

        # Synthesizer agent
        agents["synthesizer"] = LangChainSynthesizerAgent(
            llm_manager=self.llm_manager, summarizer_agent=agents["summarizer"]
        )

        # TODO: Add more agents as needed
        # agents["qa"] = QAAgent(...)
        # agents["hypothesis"] = HypothesisAgent(...)

        return agents

    async def process_request(
        self, agent_name: str, input_data: Dict[str, Any], use_memory: bool = False
    ) -> AgentResponse:
        """Process a request through specified agent"""

        if agent_name not in self.agents:
            return AgentResponse(
                success=False, data=None, error=f"Unknown agent: {agent_name}"
            )

        # Add conversation context if enabled
        if use_memory:
            memory_variables = self.conversation_memory.load_memory_variables({})
            input_data["chat_history"] = memory_variables.get("chat_history", [])

        # Process through agent
        agent = self.agents[agent_name]
        response = await agent.process(input_data)

        # Update conversation memory if enabled
        if use_memory and response.success:
            self.conversation_memory.save_context(
                {"input": input_data.get("query", "")},
                {
                    "output": response.data.get("summary", "")
                    or response.data.get("synthesis", "")
                },
            )

        # Track request
        self._track_request(agent_name, input_data, response)

        return response

    async def process_pipeline(
        self, query: str, pipeline: List[Dict[str, Any]], **kwargs
    ) -> Dict[str, Any]:
        """
        Process query through a pipeline of agents

        Pipeline format:
        [
            {"agent": "summarizer", "params": {"summary_type": "findings"}},
            {"agent": "synthesizer", "params": {"analysis_types": ["consensus", "themes"]}}
        ]
        """
        results = {}
        current_data = {"query": query, **kwargs}

        for step in pipeline:
            agent_name = step["agent"]
            params = step.get("params", {})

            # Merge params with current data
            step_input = {**current_data, **params}

            # Process through agent
            response = await self.process_request(agent_name, step_input)

            if not response.success:
                return {
                    "success": False,
                    "error": f"Pipeline failed at {agent_name}: {response.error}",
                    "completed_steps": list(results.keys()),
                    "partial_results": results,
                }

            # Store result
            results[agent_name] = response.data

            # Update data for next step
            if agent_name == "summarizer":
                current_data.update(
                    {
                        "summary_data": response.data,
                        "sources": response.data.get("sources", []),
                    }
                )
            elif agent_name == "synthesizer":
                current_data.update({"synthesis_data": response.data})

        return {
            "success": True,
            "results": results,
            "pipeline": [step["agent"] for step in pipeline],
            "query": query,
            "model_used": self.llm_manager.model_name,
        }

    def switch_model(self, model_name: str, use_8bit: bool = True):
        """Switch to a different LLM model"""
        self.llm_manager = LLMManager(model_name=model_name, use_8bit=use_8bit)

        # Reinitialize agents with new model
        self.agents = self._initialize_agents()

        # Update conversation memory
        self.conversation_memory = ConversationSummaryMemory(
            llm=self.llm_manager.get_langchain_llm(),
            memory_key="chat_history",
            return_messages=True,
        )

    def _track_request(
        self, agent_name: str, input_data: Dict[str, Any], response: AgentResponse
    ):
        """Track request for analytics"""
        self.request_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "query": input_data.get("query", ""),
                "success": response.success,
                "processing_time": response.processing_time,
                "model": self.llm_manager.model_name,
                "error": response.error if not response.success else None,
            }
        )

    def get_agent_stats(self) -> Dict[str, Any]:
        """Get statistics for all agents"""
        stats = {
            "agents": {},
            "model": self.llm_manager.model_name,
            "total_requests": len(self.request_history),
            "success_rate": (
                sum(1 for r in self.request_history if r["success"])
                / len(self.request_history)
                if self.request_history
                else 0
            ),
        }

        for name, agent in self.agents.items():
            stats["agents"][name] = agent.stats

        return stats

    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory.clear()
