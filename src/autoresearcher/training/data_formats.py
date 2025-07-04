"""
Data Format Templates for AutoResearcher Training Pipeline

This module provides standardized templates for different training data formats
including QA, instruction-following, summarization, and synthesis tasks.
"""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class FormattedSample:
    """Standard format for processed training samples"""

    input_ids: List[int]
    attention_mask: List[int]
    labels: List[int]

    # Original text for debugging
    formatted_text: str
    original_instruction: str
    original_input: str
    original_output: str

    # Metadata
    task_type: str
    source: str
    metadata: Dict[str, Any] = None


class BaseDataFormatter(ABC):
    """Abstract base class for data formatters"""

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Special tokens
        self.pad_token = tokenizer.pad_token or tokenizer.eos_token
        self.eos_token = tokenizer.eos_token
        self.bos_token = getattr(tokenizer, "bos_token", "")

    @abstractmethod
    def format_sample(
        self, instruction: str, input_text: str, output_text: str, **kwargs
    ) -> str:
        """
        Format a single sample according to the template

        # TODO: Implement template-specific formatting
        # Example: Apply instruction template, add special tokens
        # Use: String formatting, token management
        # Input: instruction, input_text, output_text (str)
        # Output: Formatted string ready for tokenization
        # Note: Handle empty inputs and special characters
        """
        pass

    def tokenize_sample(self, formatted_text: str, **kwargs) -> FormattedSample:
        """
        Tokenize formatted sample for training

        Args:
            formatted_text: Formatted text from format_sample()
            **kwargs: Additional metadata

        Returns:
            FormattedSample with tokenized data
        """
        # TODO: Implement tokenization with proper label handling
        # Example: Tokenize input, create labels for loss calculation
        # Use: tokenizer.encode, attention mask creation
        # Input: formatted_text (str)
        # Output: FormattedSample with input_ids, labels, attention_mask
        # Note: Handle truncation, padding, and label masking for instruction parts

        # Tokenize the text
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # For causal LM, labels are the same as input_ids
        # but we mask the instruction/input part to only compute loss on outputs
        labels = input_ids.copy()

        # TODO: Implement proper label masking
        # Example: Find output start position, mask instruction tokens
        # This is a simplified version - should mask instruction tokens

        return FormattedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            formatted_text=formatted_text,
            original_instruction=kwargs.get("instruction", ""),
            original_input=kwargs.get("input", ""),
            original_output=kwargs.get("output", ""),
            task_type=kwargs.get("task_type", "unknown"),
            source=kwargs.get("source", "unknown"),
            metadata=kwargs.get("metadata", {}),
        )

    def find_output_start_position(self, text: str, output_text: str) -> int:
        """Find where the output text starts in the formatted text"""
        # TODO: Implement robust output position detection
        # Example: Use response templates, pattern matching
        # Use: String search, regex matching
        # Input: text (str), output_text (str)
        # Output: int position where output starts
        # Note: Handle multiple occurrences and edge cases

        # Simple implementation - find output text in formatted text
        output_start = text.find(output_text)
        if output_start == -1:
            # Fallback: assume output is in the last 1/3 of the text
            return len(text) * 2 // 3
        return output_start


class AlpacaFormatter(BaseDataFormatter):
    """Alpaca-style instruction formatting"""

    def __init__(self, tokenizer, max_length: int = 2048):
        super().__init__(tokenizer, max_length)
        self.template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    def format_sample(
        self, instruction: str, input_text: str, output_text: str, **kwargs
    ) -> str:
        """Format sample using Alpaca template"""
        # TODO: Implement Alpaca-specific formatting rules
        # Example: Handle empty inputs, clean text, apply template
        # Use: String formatting, text cleaning
        # Input: instruction, input_text, output_text (str)
        # Output: Formatted string following Alpaca format
        # Note: Handle cases where input is empty

        # Clean inputs
        instruction = instruction.strip()
        input_text = input_text.strip()
        output_text = output_text.strip()

        # Handle empty input
        if not input_text:
            template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""
            return template.format(instruction=instruction, output=output_text)

        return self.template.format(
            instruction=instruction, input=input_text, output=output_text
        )

    def tokenize_sample(self, formatted_text: str, **kwargs) -> FormattedSample:
        """Tokenize with Alpaca-specific label masking"""
        # TODO: Implement Alpaca-specific label masking
        # Example: Mask everything before "### Response:"
        # Use: Token-level masking, response template detection
        # Input: formatted_text (str)
        # Output: FormattedSample with properly masked labels
        # Note: Only compute loss on the response part

        # Find response start
        response_start = formatted_text.find("### Response:")
        if response_start != -1:
            response_start += len("### Response:\n")

        # Tokenize full text
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.copy()

        # Mask labels before response (set to -100 to ignore in loss)
        if response_start != -1:
            # Tokenize text up to response to find mask boundary
            pre_response = formatted_text[:response_start]
            pre_response_tokens = self.tokenizer(
                pre_response, add_special_tokens=False
            )["input_ids"]
            mask_length = len(pre_response_tokens)

            # Mask the instruction part
            for i in range(min(mask_length, len(labels))):
                labels[i] = -100

        return FormattedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            formatted_text=formatted_text,
            original_instruction=kwargs.get("instruction", ""),
            original_input=kwargs.get("input", ""),
            original_output=kwargs.get("output", ""),
            task_type=kwargs.get("task_type", "unknown"),
            source=kwargs.get("source", "unknown"),
            metadata=kwargs.get("metadata", {}),
        )


class MistralFormatter(BaseDataFormatter):
    """Mistral-specific instruction formatting with chat templates"""

    def __init__(self, tokenizer, max_length: int = 2048):
        super().__init__(tokenizer, max_length)

    def format_sample(
        self, instruction: str, input_text: str, output_text: str, **kwargs
    ) -> str:
        """Format sample using Mistral chat template"""
        # TODO: Implement Mistral-specific formatting
        # Example: Use [INST] tags, handle system messages
        # Use: Mistral chat template, proper token formatting
        # Input: instruction, input_text, output_text (str)
        # Output: Formatted string with Mistral chat format
        # Note: Follow Mistral's official chat template

        # Combine instruction and input for Mistral format
        if input_text.strip():
            user_message = f"{instruction}\n\n{input_text}"
        else:
            user_message = instruction

        # Mistral chat format
        formatted = f"<s>[INST] {user_message} [/INST] {output_text}</s>"

        return formatted

    def tokenize_sample(self, formatted_text: str, **kwargs) -> FormattedSample:
        """Tokenize with Mistral-specific label masking"""
        # TODO: Implement Mistral-specific label masking
        # Example: Mask everything before [/INST]
        # Use: Mistral tokenizer features, proper chat template handling
        # Input: formatted_text (str)
        # Output: FormattedSample with Mistral-appropriate masking
        # Note: Use Mistral's recommended training approach

        # Find instruction end
        inst_end = formatted_text.find("[/INST]")
        if inst_end != -1:
            inst_end += len("[/INST] ")

        # Tokenize
        tokenized = self.tokenizer(
            formatted_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.copy()

        # Mask instruction part
        if inst_end != -1:
            pre_response = formatted_text[:inst_end]
            pre_response_tokens = self.tokenizer(
                pre_response, add_special_tokens=False
            )["input_ids"]
            mask_length = len(pre_response_tokens)

            for i in range(min(mask_length, len(labels))):
                labels[i] = -100

        return FormattedSample(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            formatted_text=formatted_text,
            original_instruction=kwargs.get("instruction", ""),
            original_input=kwargs.get("input", ""),
            original_output=kwargs.get("output", ""),
            task_type=kwargs.get("task_type", "unknown"),
            source=kwargs.get("source", "unknown"),
            metadata=kwargs.get("metadata", {}),
        )


class QAFormatter(BaseDataFormatter):
    """Question-Answering specific formatting"""

    def format_sample(
        self, instruction: str, input_text: str, output_text: str, **kwargs
    ) -> str:
        """Format QA sample with context and question structure"""
        # TODO: Implement QA-specific formatting
        # Example: Separate question and context, add QA markers
        # Use: QA template, context windowing
        # Input: instruction (question), input_text (context), output_text (answer)
        # Output: Formatted QA string
        # Note: Handle long contexts and multiple choice questions

        # Extract question from instruction or input
        question = instruction
        context = input_text
        answer = output_text

        # QA format template
        template = """Context: {context}

Question: {question}

Answer: {answer}"""

        return template.format(context=context, question=question, answer=answer)


class SummarizationFormatter(BaseDataFormatter):
    """Summarization task formatting"""

    def format_sample(
        self, instruction: str, input_text: str, output_text: str, **kwargs
    ) -> str:
        """Format summarization sample"""
        # TODO: Implement summarization-specific formatting
        # Example: Add summarization markers, handle document structure
        # Use: Summarization templates, length indicators
        # Input: instruction (summarization task), input_text (document), output_text (summary)
        # Output: Formatted summarization string
        # Note: Handle different summary types (abstract, extractive, etc.)

        template = """Document: {document}

Task: {task}

Summary: {summary}"""

        return template.format(
            document=input_text, task=instruction, summary=output_text
        )


class SynthesisFormatter(BaseDataFormatter):
    """Multi-document synthesis formatting"""

    def format_sample(
        self, instruction: str, input_text: str, output_text: str, **kwargs
    ) -> str:
        """Format synthesis sample with multiple documents"""
        # TODO: Implement synthesis-specific formatting
        # Example: Handle multiple documents, add synthesis markers
        # Use: Document separation, synthesis templates
        # Input: instruction (synthesis task), input_text (multiple docs), output_text (synthesis)
        # Output: Formatted synthesis string
        # Note: Handle document boundaries and relationship indicators

        template = """Research Papers:
{documents}

Synthesis Task: {task}

Analysis: {synthesis}"""

        return template.format(
            documents=input_text, task=instruction, synthesis=output_text
        )


class MedicalQAFormatter(BaseDataFormatter):
    """Medical domain-specific QA formatting"""

    def format_sample(
        self, instruction: str, input_text: str, output_text: str, **kwargs
    ) -> str:
        """Format medical QA with domain-specific structure"""
        # TODO: Implement medical QA formatting
        # Example: Add medical context markers, evidence requirements
        # Use: Medical QA templates, evidence-based structure
        # Input: instruction (medical question), input_text (clinical context), output_text (medical answer)
        # Output: Formatted medical QA string
        # Note: Include confidence indicators and evidence citations

        template = """Medical Question: {question}

Clinical Context: {context}

Medical Answer: {answer}"""

        return template.format(
            question=instruction, context=input_text, answer=output_text
        )


class DataFormatManager:
    """Manager class for handling different data formats"""

    def __init__(self, tokenizer, format_type: str = "alpaca", max_length: int = 2048):
        self.tokenizer = tokenizer
        self.format_type = format_type
        self.max_length = max_length

        # Initialize formatter based on type
        self.formatter = self._create_formatter()

    def _create_formatter(self) -> BaseDataFormatter:
        """Create appropriate formatter based on format type"""
        # TODO: Implement formatter factory pattern
        # Example: Return appropriate formatter class based on format_type
        # Use: Factory pattern, formatter registry
        # Input: self.format_type (str)
        # Output: BaseDataFormatter instance
        # Note: Support dynamic formatter loading and custom formatters

        formatters = {
            "alpaca": AlpacaFormatter,
            "mistral": MistralFormatter,
            "qa": QAFormatter,
            "summarization": SummarizationFormatter,
            "synthesis": SynthesisFormatter,
            "medical_qa": MedicalQAFormatter,
        }

        if self.format_type not in formatters:
            raise ValueError(
                f"Unknown format type: {self.format_type}. Available: {list(formatters.keys())}"
            )

        return formatters[self.format_type](self.tokenizer, self.max_length)

    def format_dataset(self, samples: List[Dict[str, Any]]) -> List[FormattedSample]:
        """Format a list of samples"""
        # TODO: Implement batch formatting with progress tracking
        # Example: Process samples in batches, handle errors gracefully
        # Use: tqdm for progress, error collection
        # Input: samples (List[Dict])
        # Output: List[FormattedSample]
        # Note: Handle malformed samples and provide error statistics

        formatted_samples = []
        errors = []

        for i, sample in enumerate(samples):
            try:
                # Extract required fields
                instruction = sample.get("instruction", "")
                input_text = sample.get("input", "")
                output_text = sample.get("output", "")

                # Format and tokenize
                formatted_text = self.formatter.format_sample(
                    instruction, input_text, output_text
                )
                formatted_sample = self.formatter.tokenize_sample(
                    formatted_text,
                    instruction=instruction,
                    input=input_text,
                    output=output_text,
                    task_type=sample.get("task_type", "unknown"),
                    source=sample.get("source", "unknown"),
                    metadata=sample.get("metadata", {}),
                )

                formatted_samples.append(formatted_sample)

            except Exception as e:
                errors.append(f"Sample {i}: {str(e)}")
                continue

        if errors:
            print(f"Formatting errors: {len(errors)}")
            for error in errors[:5]:  # Show first 5 errors
                print(f"  {error}")

        return formatted_samples

    def get_data_collator(self):
        """Get appropriate data collator for the format"""
        # TODO: Implement format-specific data collation
        # Example: Return DataCollatorForLanguageModeling or custom collator
        # Use: transformers.DataCollator classes
        # Input: None
        # Output: Data collator instance
        # Note: Handle different model types and training objectives

        from transformers import DataCollatorForLanguageModeling

        return DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            pad_to_multiple_of=8 if self.tokenizer.pad_token_id is not None else None,
        )


def create_format_templates() -> Dict[str, str]:
    """Create a registry of format templates for reference"""
    # TODO: Implement comprehensive template registry
    # Example: Include all supported formats with examples
    # Use: Template strings, format documentation
    # Input: None
    # Output: Dict[str, str] with format names and template strings
    # Note: Include usage examples and parameter descriptions

    templates = {
        "alpaca": """### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}""",
        "mistral": "<s>[INST] {instruction} {input} [/INST] {output}</s>",
        "qa": """Question: {instruction}
Context: {input}
Answer: {output}""",
        "medical_qa": """Medical Question: {instruction}
Clinical Context: {input}
Evidence-based Answer: {output}""",
        "summarization": """Document: {input}
Summary Task: {instruction}
Summary: {output}""",
        "synthesis": """Research Papers: {input}
Synthesis Task: {instruction}
Comprehensive Analysis: {output}""",
    }

    return templates


def validate_formatted_sample(sample: FormattedSample) -> List[str]:
    """Validate a formatted sample for common issues"""
    # TODO: Implement comprehensive sample validation
    # Example: Check token counts, label consistency, special token usage
    # Use: Statistical checks, format verification
    # Input: sample (FormattedSample)
    # Output: List[str] with validation issues
    # Note: Check for truncation, padding issues, label masking correctness

    issues = []

    # Check lengths
    if len(sample.input_ids) != len(sample.attention_mask):
        issues.append("input_ids and attention_mask length mismatch")

    if len(sample.input_ids) != len(sample.labels):
        issues.append("input_ids and labels length mismatch")

    # Check for all padding
    if all(token == 0 for token in sample.input_ids):  # Assuming 0 is pad token
        issues.append("Sample is all padding tokens")

    # Check label masking
    non_masked_labels = [l for l in sample.labels if l != -100]
    if len(non_masked_labels) == 0:
        issues.append("All labels are masked")

    # Check text truncation
    if len(sample.formatted_text) > 10000:  # Arbitrary threshold
        issues.append("Formatted text is very long, may be truncated")

    return issues


# Test function
def test_data_formats():
    """Test different data formatters"""
    print("Testing data format templates...")

    # Mock tokenizer for testing
    class MockTokenizer:
        def __init__(self):
            self.pad_token = "<pad>"
            self.eos_token = "</s>"
            self.pad_token_id = 0

        def __call__(self, text, **kwargs):
            # Simple word-based tokenization for testing
            tokens = text.split()[: kwargs.get("max_length", 100)]
            input_ids = list(range(len(tokens)))
            attention_mask = [1] * len(tokens)

            # Pad to max_length
            max_len = kwargs.get("max_length", 100)
            while len(input_ids) < max_len:
                input_ids.append(0)
                attention_mask.append(0)

            return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer = MockTokenizer()

    # Test sample data
    sample_data = {
        "instruction": "Summarize the key findings from this medical research",
        "input": "This study examined the effectiveness of metformin in treating type 2 diabetes...",
        "output": "Metformin shows 70-80% efficacy in glycemic control with minimal side effects.",
        "task_type": "summarization",
        "source": "test",
    }

    # Test different formatters
    formats = ["alpaca", "mistral", "qa", "summarization"]

    for format_type in formats:
        try:
            manager = DataFormatManager(tokenizer, format_type, max_length=100)
            formatted_samples = manager.format_dataset([sample_data])

            if formatted_samples:
                sample = formatted_samples[0]
                issues = validate_formatted_sample(sample)
                print(f"{format_type}: {len(issues)} validation issues")
            else:
                print(f"{format_type}: Failed to format sample")

        except Exception as e:
            print(f"{format_type}: Error - {e}")

    print("Data format testing completed!")


if __name__ == "__main__":
    test_data_formats()
