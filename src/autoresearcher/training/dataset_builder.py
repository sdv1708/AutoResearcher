"""
Dataset Builder Module for AutoResearcher Training Pipeline

This module provides flexible dataset builders for creating instruction-following
datasets from medical literature and existing QA datasets.
"""

import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class DatasetSample:
    """Standard format for training samples"""

    instruction: str
    input: str
    output: str
    task_type: str  # 'qa', 'summarization', 'synthesis'
    source: str  # 'pubmedqa', 'bioasq', 'documents', etc.
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
            "task_type": self.task_type,
            "source": self.source,
            "metadata": self.metadata or {},
        }


class BaseDatasetBuilder(ABC):
    """Abstract base class for dataset builders"""

    def __init__(self, output_dir: str = "./data/training", cache_dir: str = "./cache"):
        self.output_dir = Path(output_dir)
        self.cache_dir = Path(cache_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.samples: List[DatasetSample] = []
        self.statistics: Dict[str, Any] = {}

    @abstractmethod
    def load_raw_data(self) -> Any:
        """
        Load raw data from source

        # TODO: Implement using appropriate data loading method
        # Example: dataset = load_dataset('dataset_name', cache_dir=self.cache_dir)
        # Input: None (uses self.cache_dir for caching)
        # Output: Raw dataset object or data structure
        # Note: Handle network failures and cache validation
        """
        pass

    @abstractmethod
    def process_samples(self, raw_data: Any) -> List[DatasetSample]:
        """
        Convert raw data to standardized DatasetSample format

        # TODO: Implement data processing logic
        # Example: samples = [DatasetSample(instruction=..., input=..., output=...) for item in raw_data]
        # Input: raw_data from load_raw_data()
        # Output: List[DatasetSample] with processed samples
        # Note: Handle malformed data and missing fields
        """
        pass

    def create_splits(
        self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42
    ) -> Tuple[List[DatasetSample], List[DatasetSample], List[DatasetSample]]:
        """
        Create train/validation/test splits

        Args:
            test_size: Fraction of data for test set
            val_size: Fraction of remaining data for validation set
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_samples, val_samples, test_samples)
        """
        if not self.samples:
            raise ValueError("No samples available. Call process_samples() first.")

        # TODO: Implement stratified splitting if needed
        # Example: train_test_split(self.samples, test_size=test_size, random_state=random_state, stratify=labels)
        # Input: self.samples (List[DatasetSample])
        # Output: Tuple of train, val, test splits
        # Note: Consider stratification by task_type or source

        # First split: separate test set
        train_val, test = train_test_split(
            self.samples, test_size=test_size, random_state=random_state
        )

        # Second split: separate validation from training
        if val_size > 0:
            val_ratio = val_size / (1 - test_size)
            train, val = train_test_split(
                train_val, test_size=val_ratio, random_state=random_state
            )
        else:
            train, val = train_val, []

        logger.info(
            f"Created splits: train={len(train)}, val={len(val)}, test={len(test)}"
        )
        return train, val, test

    def save_dataset(
        self,
        train_samples: List[DatasetSample],
        val_samples: List[DatasetSample],
        test_samples: List[DatasetSample],
        dataset_name: str,
    ) -> str:
        """
        Save dataset in HuggingFace format

        Args:
            train_samples: Training samples
            val_samples: Validation samples
            test_samples: Test samples
            dataset_name: Name for saved dataset

        Returns:
            Path to saved dataset directory
        """
        # TODO: Implement HuggingFace dataset saving
        # Example:
        # train_dict = [sample.to_dict() for sample in train_samples]
        # dataset_dict = DatasetDict({
        #     'train': Dataset.from_list(train_dict),
        #     'validation': Dataset.from_list(val_dict),
        #     'test': Dataset.from_list(test_dict)
        # })
        # dataset_dict.save_to_disk(save_path)
        # Input: List[DatasetSample] for each split
        # Output: str path to saved dataset
        # Note: Handle disk space and permission errors

        save_path = self.output_dir / dataset_name

        # Convert samples to dictionaries
        train_dict = [sample.to_dict() for sample in train_samples]
        val_dict = [sample.to_dict() for sample in val_samples]
        test_dict = [sample.to_dict() for sample in test_samples]

        # Create HuggingFace dataset
        dataset_dict = DatasetDict(
            {
                "train": Dataset.from_list(train_dict),
                "validation": Dataset.from_list(val_dict),
                "test": Dataset.from_list(test_dict),
            }
        )

        # Save to disk
        dataset_dict.save_to_disk(str(save_path))

        # Save metadata
        metadata = {
            "dataset_name": dataset_name,
            "total_samples": len(self.samples),
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
            "statistics": self.statistics,
        }

        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved dataset to {save_path}")
        return str(save_path)

    def get_statistics(self) -> Dict[str, Any]:
        """
        Calculate dataset statistics

        Returns:
            Dictionary with dataset metrics
        """
        if not self.samples:
            return {}

        # TODO: Implement comprehensive statistics calculation
        # Example: Calculate token counts, answer lengths, task distribution
        # Use: collections.Counter, numpy statistical functions
        # Input: self.samples (List[DatasetSample])
        # Output: Dict with keys like 'avg_input_length', 'task_distribution', etc.
        # Note: Handle empty strings and None values

        stats = {
            "total_samples": len(self.samples),
            "task_types": {},
            "sources": {},
            "avg_input_length": 0,
            "avg_output_length": 0,
            "avg_instruction_length": 0,
        }

        # Count task types and sources
        for sample in self.samples:
            # Task type distribution
            if sample.task_type in stats["task_types"]:
                stats["task_types"][sample.task_type] += 1
            else:
                stats["task_types"][sample.task_type] = 1

            # Source distribution
            if sample.source in stats["sources"]:
                stats["sources"][sample.source] += 1
            else:
                stats["sources"][sample.source] = 1

        # Calculate average lengths
        if self.samples:
            total_input_len = sum(len(s.input.split()) for s in self.samples)
            total_output_len = sum(len(s.output.split()) for s in self.samples)
            total_instruction_len = sum(
                len(s.instruction.split()) for s in self.samples
            )

            stats["avg_input_length"] = total_input_len / len(self.samples)
            stats["avg_output_length"] = total_output_len / len(self.samples)
            stats["avg_instruction_length"] = total_instruction_len / len(self.samples)

        self.statistics = stats
        return stats

    def build_dataset(self, dataset_name: str) -> str:
        """
        Complete dataset building pipeline

        Args:
            dataset_name: Name for the output dataset

        Returns:
            Path to saved dataset
        """
        logger.info(f"Building dataset: {dataset_name}")

        # Load and process data
        raw_data = self.load_raw_data()
        self.samples = self.process_samples(raw_data)

        # Calculate statistics
        self.get_statistics()

        # Create splits
        train, val, test = self.create_splits()

        # Save dataset
        return self.save_dataset(train, val, test, dataset_name)


class PubMedQADatasetBuilder(BaseDatasetBuilder):
    """Dataset builder for PubMedQA dataset"""

    def __init__(self, subset: str = "pqa_labeled", **kwargs):
        super().__init__(**kwargs)
        self.subset = subset

    def load_raw_data(self) -> Any:
        """Load PubMedQA dataset from HuggingFace"""
        # TODO: Implement PubMedQA dataset loading
        # Example: dataset = load_dataset('pubmed_qa', self.subset, cache_dir=self.cache_dir)
        # Input: self.subset ('pqa_labeled', 'pqa_unlabeled', 'pqa_artificial')
        # Output: HuggingFace dataset object
        # Note: Handle different subset formats and missing data

        try:
            dataset = load_dataset(
                "pubmed_qa", self.subset, cache_dir=str(self.cache_dir)
            )
            logger.info(f"Loaded PubMedQA dataset subset: {self.subset}")
            return dataset
        except Exception as e:
            logger.error(f"Failed to load PubMedQA dataset: {e}")
            # Return empty dataset for testing
            return {"train": []}

    def process_samples(self, raw_data: Any) -> List[DatasetSample]:
        """Convert PubMedQA samples to standard format"""
        # TODO: Implement PubMedQA sample processing
        # Example: Extract 'question', 'context', 'long_answer' fields
        # Use: Text cleaning, format standardization
        # Input: raw_data (HuggingFace dataset)
        # Output: List[DatasetSample] with QA format
        # Note: Handle different answer formats (yes/no/maybe vs long answers)

        samples = []

        # Process training split if available
        if "train" in raw_data:
            for item in tqdm(raw_data["train"], desc="Processing PubMedQA samples"):
                try:
                    # Extract fields
                    question = item.get("question", "")
                    context = " ".join(item.get("context", {}).get("contexts", []))
                    answer = item.get("long_answer", item.get("final_decision", ""))

                    # Create instruction
                    instruction = "Answer the following medical question based on the provided context."

                    # Format input
                    input_text = f"Question: {question}\n\nContext: {context}"

                    # Create sample
                    sample = DatasetSample(
                        instruction=instruction,
                        input=input_text,
                        output=answer,
                        task_type="qa",
                        source="pubmedqa",
                        metadata={"pubmed_id": item.get("pubid", "")},
                    )

                    samples.append(sample)

                except Exception as e:
                    logger.warning(f"Failed to process PubMedQA sample: {e}")
                    continue

        logger.info(f"Processed {len(samples)} PubMedQA samples")
        return samples


class BioASQDatasetBuilder(BaseDatasetBuilder):
    """Dataset builder for BioASQ dataset"""

    def __init__(self, task_type: str = "factoid", **kwargs):
        super().__init__(**kwargs)
        self.task_type = task_type  # 'factoid', 'list', 'yesno', 'summary'

    def load_raw_data(self) -> Any:
        """Load BioASQ dataset"""
        # TODO: Implement BioASQ dataset loading
        # Example: dataset = load_dataset('bioasq', self.task_type, cache_dir=self.cache_dir)
        # Input: self.task_type (str)
        # Output: Dataset object or JSON data
        # Note: BioASQ may require manual download and local loading

        # Placeholder implementation - BioASQ requires special handling
        logger.warning("BioASQ dataset loading not implemented - using placeholder")
        return {"train": []}

    def process_samples(self, raw_data: Any) -> List[DatasetSample]:
        """Convert BioASQ samples to standard format"""
        # TODO: Implement BioASQ sample processing
        # Example: Handle different question types (factoid, list, yes/no, summary)
        # Use: Question type detection, answer format standardization
        # Input: raw_data (BioASQ dataset)
        # Output: List[DatasetSample] with appropriate format
        # Note: Different question types need different processing

        samples = []

        # Process based on task type
        for item in raw_data.get("train", []):
            try:
                question = item.get("body", "")
                snippets = item.get("snippets", [])
                context = " ".join([s.get("text", "") for s in snippets])

                # Handle different answer formats
                if self.task_type == "factoid":
                    answers = item.get("exact_answer", [])
                    answer = answers[0] if answers else ""
                elif self.task_type == "yesno":
                    answer = item.get("exact_answer", "no")
                elif self.task_type == "list":
                    answers = item.get("exact_answer", [])
                    answer = "; ".join(answers)
                else:  # summary
                    answer = item.get("ideal_answer", "")

                instruction = f"Answer this {self.task_type} biomedical question based on the provided context."
                input_text = f"Question: {question}\n\nContext: {context}"

                sample = DatasetSample(
                    instruction=instruction,
                    input=input_text,
                    output=answer,
                    task_type="qa",
                    source="bioasq",
                    metadata={
                        "question_type": self.task_type,
                        "question_id": item.get("id", ""),
                    },
                )

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Failed to process BioASQ sample: {e}")
                continue

        logger.info(f"Processed {len(samples)} BioASQ samples")
        return samples


class InstructionDatasetBuilder(BaseDatasetBuilder):
    """Convert document chunks to instruction-following format"""

    def __init__(self, faiss_index_path: str, embedding_pipeline=None, **kwargs):
        super().__init__(**kwargs)
        self.faiss_index_path = faiss_index_path
        self.embedding_pipeline = embedding_pipeline
        self.documents = []

    def load_raw_data(self) -> Any:
        """Load document chunks from FAISS index"""
        # TODO: Implement FAISS index loading
        # Example: Load index and metadata using FAISSIndexManager
        # Use: from ..embeddings.faiss_index import FAISSIndexManager
        # Input: self.faiss_index_path (str)
        # Output: List of document chunks with metadata
        # Note: Handle missing index files and corrupted data

        try:
            # Load FAISS index and metadata
            # This is a placeholder - integrate with existing FAISS code
            logger.info(f"Loading documents from FAISS index: {self.faiss_index_path}")

            # TODO: Replace with actual FAISS loading
            # from ..embeddings.faiss_index import FAISSIndexManager
            # index_manager = FAISSIndexManager()
            # index_manager.load_index(self.faiss_index_path)
            # self.documents = index_manager.documents

            # Placeholder return
            return []

        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return []

    def process_samples(self, raw_data: Any) -> List[DatasetSample]:
        """Convert document chunks to instruction datasets"""
        # TODO: Implement document-to-instruction conversion
        # Example: Create summarization, QA, and synthesis tasks from chunks
        # Use: Template-based instruction generation, chunk combination
        # Input: raw_data (list of document chunks)
        # Output: List[DatasetSample] with various task types
        # Note: Generate diverse instruction types and ensure quality

        samples = []

        # Generate summarization tasks
        summarization_samples = self._create_summarization_tasks(raw_data)
        samples.extend(summarization_samples)

        # Generate QA tasks
        qa_samples = self._create_qa_tasks(raw_data)
        samples.extend(qa_samples)

        # Generate synthesis tasks
        synthesis_samples = self._create_synthesis_tasks(raw_data)
        samples.extend(synthesis_samples)

        logger.info(f"Generated {len(samples)} instruction samples from documents")
        return samples

    def _create_summarization_tasks(self, documents: List[Dict]) -> List[DatasetSample]:
        """Create summarization tasks from document chunks"""
        # TODO: Implement summarization task generation
        # Example: Use document titles/abstracts as summaries, full text as input
        # Use: Template variations, chunk combination strategies
        # Input: documents (List[Dict] with chunk data)
        # Output: List[DatasetSample] for summarization
        # Note: Ensure summary quality and appropriate length

        samples = []

        summarization_instructions = [
            "Summarize the key findings from this medical research text.",
            "Provide a concise summary of the main points in this medical document.",
            "Extract the most important information from this research paper section.",
        ]

        for doc in documents:
            try:
                # Use abstract as summary target if available
                if (
                    doc.get("chunk_type") == "abstract"
                    and len(doc.get("text", "")) > 100
                ):
                    continue  # Skip abstracts as they're already summaries

                # Create summarization task
                instruction = summarization_instructions[
                    hash(doc.get("chunk_id", "")) % len(summarization_instructions)
                ]

                sample = DatasetSample(
                    instruction=instruction,
                    input=doc.get("text", ""),
                    output=self._generate_summary(doc.get("text", "")),  # Placeholder
                    task_type="summarization",
                    source="documents",
                    metadata={
                        "document_id": doc.get("document_id", ""),
                        "chunk_type": doc.get("chunk_type", ""),
                    },
                )

                samples.append(sample)

            except Exception as e:
                logger.warning(f"Failed to create summarization task: {e}")
                continue

        return samples[:100]  # Limit for testing

    def _create_qa_tasks(self, documents: List[Dict]) -> List[DatasetSample]:
        """Create QA tasks from document chunks"""
        # TODO: Implement QA task generation
        # Example: Generate questions about key concepts, use named entity recognition
        # Use: Question templates, entity extraction, answer span identification
        # Input: documents (List[Dict])
        # Output: List[DatasetSample] for QA
        # Note: Ensure questions are answerable and answers are accurate

        samples = []

        qa_instruction = (
            "Answer the following question based on the provided medical research text."
        )

        # TODO: Implement question generation logic
        # This is a placeholder implementation

        return samples[:50]  # Limit for testing

    def _create_synthesis_tasks(self, documents: List[Dict]) -> List[DatasetSample]:
        """Create synthesis tasks from multiple document chunks"""
        # TODO: Implement synthesis task generation
        # Example: Combine related chunks, create comparison tasks
        # Use: Semantic similarity, topic grouping, synthesis templates
        # Input: documents (List[Dict])
        # Output: List[DatasetSample] for synthesis
        # Note: Ensure documents are related and synthesis is meaningful

        samples = []

        synthesis_instruction = "Synthesize information from these medical research texts to provide a comprehensive analysis."

        # TODO: Group related documents and create synthesis tasks
        # This is a placeholder implementation

        return samples[:25]  # Limit for testing

    def _generate_summary(self, text: str) -> str:
        """Generate summary for text (placeholder)"""
        # TODO: Implement actual summarization
        # Example: Use existing summarizer agent or extractive methods
        # Use: Text ranking, sentence extraction, length control
        # Input: text (str)
        # Output: summary (str)
        # Note: Maintain medical accuracy and key information

        # Placeholder - return first few sentences
        sentences = text.split(".")[:3]
        return ". ".join(sentences) + "." if sentences else text[:200]


class DatasetMerger:
    """Merge multiple datasets with balancing and deduplication"""

    def __init__(self, output_dir: str = "./data/training/merged"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def merge_datasets(
        self,
        dataset_paths: List[str],
        weights: Optional[List[float]] = None,
        output_name: str = "merged_dataset",
    ) -> str:
        """
        Merge multiple datasets with optional weighting

        Args:
            dataset_paths: List of paths to saved datasets
            weights: Optional weights for each dataset
            output_name: Name for merged dataset

        Returns:
            Path to merged dataset
        """
        # TODO: Implement dataset merging logic
        # Example: Load datasets, apply weights, deduplicate, shuffle
        # Use: datasets.concatenate_datasets, hash-based deduplication
        # Input: dataset_paths (List[str]), weights (Optional[List[float]])
        # Output: str path to merged dataset
        # Note: Handle memory efficiently for large datasets, preserve splits

        logger.info(f"Merging {len(dataset_paths)} datasets")

        all_datasets = []

        for i, path in enumerate(dataset_paths):
            try:
                # Load dataset
                dataset = DatasetDict.load_from_disk(path)

                # Apply weight if provided
                if weights and i < len(weights):
                    weight = weights[i]
                    # Sample dataset according to weight
                    for split in dataset:
                        size = int(len(dataset[split]) * weight)
                        dataset[split] = dataset[split].select(
                            range(min(size, len(dataset[split])))
                        )

                all_datasets.append(dataset)
                logger.info(f"Loaded dataset from {path}")

            except Exception as e:
                logger.error(f"Failed to load dataset from {path}: {e}")
                continue

        if not all_datasets:
            raise ValueError("No datasets could be loaded")

        # Merge datasets
        merged = {}
        for split in all_datasets[0].keys():
            split_datasets = []
            for dataset in all_datasets:
                if split in dataset:
                    split_datasets.append(dataset[split])

            if split_datasets:
                # TODO: Implement deduplication here
                # Example: Use hash-based deduplication on input text
                merged[split] = concatenate_datasets(split_datasets)

        merged_dataset = DatasetDict(merged)

        # Save merged dataset
        output_path = self.output_dir / output_name
        merged_dataset.save_to_disk(str(output_path))

        logger.info(f"Merged dataset saved to {output_path}")
        return str(output_path)

    def deduplicate_dataset(
        self, dataset: Dataset, text_column: str = "input"
    ) -> Dataset:
        """Remove duplicate samples based on text content"""
        # TODO: Implement deduplication logic
        # Example: Use text hashing, fuzzy matching for near-duplicates
        # Use: hashlib, difflib, or sentence-transformers for semantic similarity
        # Input: dataset (Dataset), text_column (str)
        # Output: Dataset without duplicates
        # Note: Consider both exact and semantic duplicates

        seen_hashes = set()
        unique_indices = []

        for i, sample in enumerate(dataset):
            text = sample.get(text_column, "")
            text_hash = hash(text.lower().strip())

            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique_indices.append(i)

        logger.info(f"Removed {len(dataset) - len(unique_indices)} duplicates")
        return dataset.select(unique_indices)


# Convenience function for testing
def test_dataset_builders():
    """Test all dataset builders with small samples"""
    # TODO: Implement comprehensive testing
    # Example: Create small test datasets, verify formats, check statistics
    # Use: assert statements, sample data validation
    # Input: None
    # Output: Test results and validation messages
    # Note: Test with minimal data to avoid long downloads

    logger.info("Testing dataset builders...")

    # Test PubMedQA builder
    try:
        pubmedqa_builder = PubMedQADatasetBuilder()
        # Use a subset for testing
        logger.info("PubMedQA builder created successfully")
    except Exception as e:
        logger.error(f"PubMedQA builder test failed: {e}")

    # Test BioASQ builder
    try:
        bioasq_builder = BioASQDatasetBuilder()
        logger.info("BioASQ builder created successfully")
    except Exception as e:
        logger.error(f"BioASQ builder test failed: {e}")

    # Test Instruction builder
    try:
        instruction_builder = InstructionDatasetBuilder("./data/faiss_index")
        logger.info("Instruction builder created successfully")
    except Exception as e:
        logger.error(f"Instruction builder test failed: {e}")

    # Test DatasetMerger
    try:
        merger = DatasetMerger()
        logger.info("Dataset merger created successfully")
    except Exception as e:
        logger.error(f"Dataset merger test failed: {e}")

    logger.info("Dataset builder testing completed")


if __name__ == "__main__":
    test_dataset_builders()
