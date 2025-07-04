# src/autoresearcher/llm/llm_manager.py
"""
LLM Manager for flexible model selection
Supports: Mistral, Llama 2, Falcon, BioMedLM
"""
from typing import Any, Dict, Optional

import torch
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms import HuggingFacePipeline
from langchain.llms.base import LLM
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


class LLMConfig:
    """LLM Configuration"""

    MODELS = {
        "mistral-7b": {
            "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
            "task": "text-generation",
            "temperature": 0.3,
            "max_length": 2048,
            "requires_auth": False,
        },
        "llama2-7b": {
            "model_id": "meta-llama/Llama-2-7b-chat-hf",
            "task": "text-generation",
            "temperature": 0.3,
            "max_length": 2048,
            "requires_auth": True,
        },
        "biomedlm": {
            "model_id": "stanford-crfm/BioMedLM",
            "task": "text-generation",
            "temperature": 0.3,
            "max_length": 1024,
            "requires_auth": False,
        },
        "falcon-7b": {
            "model_id": "tiiuae/falcon-7b-instruct",
            "task": "text-generation",
            "temperature": 0.3,
            "max_length": 2048,
            "requires_auth": False,
        },
    }


class CustomLLM(LLM):
    """Custom LLM wrapper for LangChain compatibility"""

    pipeline: Any
    model_name: str = "mistral-7b"

    @property
    def _llm_type(self) -> str:
        return f"custom_{self.model_name}"

    def _call(
        self,
        prompt: str,
        stop: Optional[list] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        """Call the LLM with the given prompt."""
        response = self.pipeline(
            prompt,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            top_p=0.9,
            stop_sequences=stop,
        )
        return response[0]["generated_text"].split(prompt)[-1].strip()


class LLMManager:
    """Manages LLM initialization and switching"""

    def __init__(
        self,
        model_name: str = "mistral-7b",
        use_8bit: bool = True,
        use_lora: bool = False,
        lora_weights_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.model_name = model_name
        self.use_8bit = use_8bit
        self.use_lora = use_lora
        self.lora_weights_path = lora_weights_path
        self.device = device

        # Get model config
        if model_name not in LLMConfig.MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. Choose from: {list(LLMConfig.MODELS.keys())}"
            )

        self.config = LLMConfig.MODELS[model_name]

    def get_langchain_llm(self) -> LLM:
        """Get LangChain-compatible LLM"""
        # For mock/testing
        if self.model_name == "mock":
            from langchain.llms.fake import FakeListLLM

            return FakeListLLM(
                responses=[
                    "Based on the research papers, the key findings are...",
                    "The consensus among studies shows...",
                    "The main contradictions identified are...",
                ]
            )

        # Load real model
        model_id = self.config["model_id"]

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with optimizations
        if self.use_8bit:
            from transformers import BitsAndBytesConfig

            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True, bnb_8bit_compute_dtype=torch.float16
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map=self.device,
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )

        # Load LoRA weights if specified
        if self.use_lora and self.lora_weights_path:
            model = PeftModel.from_pretrained(model, self.lora_weights_path)

        # Create pipeline
        pipe = pipeline(
            self.config["task"],
            model=model,
            tokenizer=tokenizer,
            device_map=self.device,
        )

        # Create custom LLM wrapper
        llm = CustomLLM(pipeline=pipe, model_name=self.model_name)

        return llm

    def get_prompt_template(self, template_type: str = "summary") -> str:
        """Get model-specific prompt template"""
        templates = {
            "mistral-7b": {
                "summary": """<s>[INST] You are a medical research assistant. {instruction}

Context:
{context}

Question: {question}
[/INST]

{output_format}""",
                "qa": """<s>[INST] Answer the question based on the provided medical research context.

Context: {context}

Question: {question}
[/INST]

Answer:""",
            },
            "llama2-7b": {
                "summary": """<s>[INST] <<SYS>>
You are a helpful medical research assistant that summarizes scientific papers accurately.
<</SYS>>

{instruction}

Context:
{context}

Question: {question}
[/INST]

{output_format}""",
                "qa": """<s>[INST] <<SYS>>
You are a medical expert assistant.
<</SYS>>

Context: {context}
Question: {question}
[/INST]""",
            },
            "default": {
                "summary": """{instruction}

Context:
{context}

Question: {question}

{output_format}""",
                "qa": """Context: {context}

Question: {question}

Answer:""",
            },
        }

        model_templates = templates.get(self.model_name, templates["default"])
        return model_templates.get(template_type, model_templates["summary"])
