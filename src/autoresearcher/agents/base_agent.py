# src/autoresearcher/agents/base_agent.py
"""
Base agent class using LangChain
"""
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

if TYPE_CHECKING:
    from ..llm.llm_manager import LLMManager

logger = logging.getLogger(__name__)


@dataclass
class AgentResponse:
    """Standard response format for all agents"""

    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    processing_time: float = 0.0


class AgentCallbackHandler(BaseCallbackHandler):
    """Callback handler for agent operations"""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.tokens_used = 0

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        logger.info(f"{self.agent_name} - LLM started with {len(prompts)} prompts")

    def on_llm_end(self, response: Any, **kwargs) -> None:
        logger.info(f"{self.agent_name} - LLM completed")


class BaseLangChainAgent(ABC):
    """Base class for LangChain-based agents"""

    def __init__(self, name: str, llm_manager: "LLMManager"):
        self.name = name
        self.logger = logger.getChild(name)
        self.llm_manager = llm_manager
        self.llm = llm_manager.get_langchain_llm()

        # Memory for conversation context
        self.memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        # Callbacks
        self.callbacks = [AgentCallbackHandler(name)]

        # Statistics
        self.stats = {
            "requests_processed": 0,
            "errors": 0,
            "total_processing_time": 0.0,
            "total_tokens": 0,
        }

    async def process(self, input_data: Dict[str, Any]) -> AgentResponse:
        """Process input and return response"""
        start_time = time.time()

        try:
            # Validate input
            self.validate_input(input_data)

            # Process with LangChain
            result = await self._process_with_langchain(input_data)

            # Update stats
            processing_time = time.time() - start_time
            self.stats["requests_processed"] += 1
            self.stats["total_processing_time"] += processing_time

            return AgentResponse(
                success=True,
                data=result,
                processing_time=processing_time,
                metadata={"model": self.llm_manager.model_name},
            )

        except Exception as e:
            self.logger.error(f"Error in {self.name}: {str(e)}")
            self.stats["errors"] += 1

            return AgentResponse(
                success=False,
                data=None,
                error=str(e),
                processing_time=time.time() - start_time,
            )

    @abstractmethod
    async def _process_with_langchain(self, input_data: Dict[str, Any]) -> Any:
        """Process using LangChain - implement in subclass"""
        pass

    @abstractmethod
    def validate_input(self, input_data: Dict[str, Any]):
        """Validate input data - implement in subclass"""
        pass
