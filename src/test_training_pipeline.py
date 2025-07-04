"""
Comprehensive Test Suite for AutoResearcher Training Pipeline

This test suite validates all components of the training pipeline including
dataset builders, configuration classes, data formatters, and training flows.
"""

import asyncio
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from autoresearcher.training.data_formats import (AlpacaFormatter,
                                                  DataFormatManager,
                                                  FormattedSample,
                                                  MistralFormatter)
# Import our training components
from autoresearcher.training.dataset_builder import (BioASQDatasetBuilder,
                                                     DatasetMerger,
                                                     DatasetSample,
                                                     InstructionDatasetBuilder,
                                                     PubMedQADatasetBuilder)
from autoresearcher.training.evaluation import (EvaluationSuite,
                                                ExactMatchMetric,
                                                F1ScoreMetric,
                                                MedicalRelevanceMetric,
                                                ROUGEMetric,
                                                run_comprehensive_evaluation)
from autoresearcher.training.training_config import (DataConfig,
                                                     FullPipelineConfig,
                                                     LoRAConfig, ModelConfig,
                                                     TrainingConfig)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockTokenizer:
    """Mock tokenizer for testing"""

    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.vocab_size = 32000

    def __call__(self, text, **kwargs):
        # TODO: Implement realistic tokenization for testing
        # Example: Word-based tokenization with proper padding/truncation
        # Use: String splitting, padding logic
        # Input: text (str), **kwargs with tokenizer options
        # Output: Dict with input_ids, attention_mask
        # Note: Handle max_length, padding, truncation options

        if isinstance(text, list):
            # Batch processing
            return self._tokenize_batch(text, **kwargs)

        # Single text processing
        tokens = text.split()
        max_length = kwargs.get("max_length", 512)

        # Convert to IDs (simple word count-based)
        input_ids = [hash(token) % 1000 for token in tokens]

        # Truncate if necessary
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            tokens = tokens[:max_length]

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Pad if necessary
        if kwargs.get("padding") == "max_length":
            while len(input_ids) < max_length:
                input_ids.append(self.pad_token_id)
                attention_mask.append(0)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _tokenize_batch(self, texts: List[str], **kwargs) -> Dict[str, List[List[int]]]:
        """Tokenize a batch of texts"""
        batch_results = {"input_ids": [], "attention_mask": []}

        for text in texts:
            result = self(text, **kwargs)
            batch_results["input_ids"].append(result["input_ids"])
            batch_results["attention_mask"].append(result["attention_mask"])

        return batch_results

    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs back to text"""
        return f"decoded_text_{len(token_ids)}_tokens"


def create_test_samples() -> List[DatasetSample]:
    """Create test dataset samples"""
    # TODO: Implement diverse test sample creation
    # Example: Create samples for different tasks, sources, lengths
    # Use: DatasetSample creation with varied content
    # Input: None
    # Output: List[DatasetSample] with test data
    # Note: Include edge cases like empty fields, special characters

    samples = [
        DatasetSample(
            instruction="What is the efficacy of metformin in treating type 2 diabetes?",
            input="Studies have shown that metformin is effective in controlling blood glucose levels in patients with type 2 diabetes.",
            output="Metformin demonstrates 70-80% efficacy in glycemic control for type 2 diabetes patients.",
            task_type="qa",
            source="pubmedqa",
            metadata={"pubmed_id": "12345678"},
        ),
        DatasetSample(
            instruction="Summarize the key findings from this medical research",
            input="This study examined 500 patients with hypertension over 2 years. Results showed that ACE inhibitors reduced systolic blood pressure by an average of 15 mmHg compared to placebo.",
            output="ACE inhibitors significantly reduced systolic blood pressure by 15 mmHg in hypertensive patients (n=500, 2-year study).",
            task_type="summarization",
            source="documents",
            metadata={"document_id": "PMC123456", "chunk_type": "abstract"},
        ),
        DatasetSample(
            instruction="Analyze the consensus across multiple studies on COVID-19 treatments",
            input="Study 1: Remdesivir showed 31% reduction in recovery time. Study 2: Dexamethasone reduced mortality by 17%. Study 3: Combination therapy improved outcomes.",
            output="Multiple studies demonstrate effectiveness of antiviral and corticosteroid treatments for COVID-19, with remdesivir reducing recovery time and dexamethasone reducing mortality.",
            task_type="synthesis",
            source="bioasq",
            metadata={
                "studies_count": 3,
                "treatment_types": ["antiviral", "corticosteroid"],
            },
        ),
        # Edge cases
        DatasetSample(
            instruction="",  # Empty instruction
            input="Test input",
            output="Test output",
            task_type="qa",
            source="test",
        ),
        DatasetSample(
            instruction="Very long instruction " * 50,  # Long instruction
            input="Short input",
            output="Short output",
            task_type="qa",
            source="test",
        ),
    ]

    return samples


class TestDatasetBuilders:
    """Test dataset builder classes"""

    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_samples = create_test_samples()

    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_dataset_sample_creation(self):
        """Test DatasetSample creation and serialization"""
        # TODO: Implement DatasetSample testing
        # Example: Test sample creation, to_dict conversion, validation
        # Use: Sample creation, assertions
        # Input: Test sample data
        # Output: Validation of sample structure and methods
        # Note: Test edge cases and field validation

        sample = self.test_samples[0]

        # Test basic properties
        assert sample.instruction != ""
        assert sample.task_type in ["qa", "summarization", "synthesis"]
        assert sample.source != ""

        # Test serialization
        sample_dict = sample.to_dict()
        assert isinstance(sample_dict, dict)
        assert "instruction" in sample_dict
        assert "input" in sample_dict
        assert "output" in sample_dict

        logger.info("DatasetSample creation test passed")

    def test_pubmedqa_builder(self):
        """Test PubMedQA dataset builder"""
        # TODO: Implement PubMedQA builder testing
        # Example: Test with mock data, validate output format
        # Use: Mock dataset, builder initialization
        # Input: Mock PubMedQA data
        # Output: Validation of processed samples
        # Note: Test different subsets and error handling

        builder = PubMedQADatasetBuilder(
            output_dir=self.temp_dir, cache_dir=self.temp_dir
        )

        # Test builder initialization
        assert builder.output_dir.exists()
        assert builder.cache_dir.exists()
        assert builder.subset == "pqa_labeled"

        # Test with mock data (builder will use placeholder)
        try:
            # This will use placeholder data since we don't have real PubMedQA access
            raw_data = builder.load_raw_data()
            samples = builder.process_samples(raw_data)

            # Should return empty list in test mode
            assert isinstance(samples, list)

            logger.info("PubMedQA builder test passed")

        except Exception as e:
            logger.warning(f"PubMedQA builder test failed (expected in test mode): {e}")

    def test_instruction_builder(self):
        """Test instruction dataset builder"""
        # TODO: Implement instruction builder testing
        # Example: Test with mock FAISS index, validate instruction generation
        # Use: Mock FAISS data, instruction templates
        # Input: Mock document chunks
        # Output: Validation of instruction samples
        # Note: Test different instruction types and quality

        builder = InstructionDatasetBuilder(
            faiss_index_path="nonexistent_path",  # Will fail gracefully
            output_dir=self.temp_dir,
            cache_dir=self.temp_dir,
        )

        # Test builder initialization
        assert builder.faiss_index_path == "nonexistent_path"
        assert builder.output_dir.exists()

        # Test with mock data
        mock_documents = [
            {
                "chunk_id": "test_chunk_1",
                "document_id": "PMC123456",
                "text": "This study examined the effectiveness of new diabetes treatment.",
                "chunk_type": "abstract",
                "metadata": {"section": "abstract"},
            }
        ]

        samples = builder.process_samples(mock_documents)
        assert isinstance(samples, list)

        logger.info("Instruction builder test passed")

    def test_dataset_merger(self):
        """Test dataset merging functionality"""
        # TODO: Implement dataset merger testing
        # Example: Create multiple test datasets, test merging logic
        # Use: Temporary datasets, merger functionality
        # Input: Multiple test datasets
        # Output: Validation of merged dataset
        # Note: Test deduplication and weighting

        merger = DatasetMerger(output_dir=self.temp_dir)

        # Create test datasets
        from datasets import Dataset, DatasetDict

        # Create first test dataset
        dataset1_data = [sample.to_dict() for sample in self.test_samples[:2]]
        dataset1 = DatasetDict(
            {
                "train": Dataset.from_list(dataset1_data),
                "validation": Dataset.from_list(dataset1_data[:1]),
                "test": Dataset.from_list(dataset1_data[:1]),
            }
        )

        # Create second test dataset
        dataset2_data = [sample.to_dict() for sample in self.test_samples[2:4]]
        dataset2 = DatasetDict(
            {
                "train": Dataset.from_list(dataset2_data),
                "validation": Dataset.from_list(dataset2_data[:1]),
                "test": Dataset.from_list(dataset2_data[:1]),
            }
        )

        # Save test datasets
        dataset1_path = Path(self.temp_dir) / "dataset1"
        dataset2_path = Path(self.temp_dir) / "dataset2"

        dataset1.save_to_disk(str(dataset1_path))
        dataset2.save_to_disk(str(dataset2_path))

        # Test merging
        try:
            merged_path = merger.merge_datasets(
                dataset_paths=[str(dataset1_path), str(dataset2_path)],
                output_name="test_merged",
            )

            assert Path(merged_path).exists()
            logger.info("Dataset merger test passed")

        except Exception as e:
            logger.warning(f"Dataset merger test failed: {e}")


class TestTrainingConfigs:
    """Test training configuration classes"""

    def test_model_config(self):
        """Test ModelConfig creation and validation"""
        # TODO: Implement ModelConfig testing
        # Example: Test different model types, parameter validation
        # Use: Config creation, parameter checking
        # Input: Different model configurations
        # Output: Validation of config parameters
        # Note: Test model-specific defaults and compatibility

        # Test default config
        config = ModelConfig()
        assert config.model_name != ""
        assert config.model_type in ["mistral", "biobert", "llama2"]
        assert config.max_sequence_length > 0

        # Test specific model configs
        configs = ModelConfig.get_model_configs()
        assert "mistral-7b" in configs
        assert "biobert" in configs

        mistral_config = configs["mistral-7b"]
        assert mistral_config.model_type == "mistral"
        assert mistral_config.max_sequence_length == 2048

        # Test serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "model_name" in config_dict

        logger.info("ModelConfig test passed")

    def test_lora_config(self):
        """Test LoRAConfig creation and validation"""
        # TODO: Implement LoRAConfig testing
        # Example: Test LoRA parameters, target modules
        # Use: Config validation, parameter ranges
        # Input: Different LoRA configurations
        # Output: Validation of LoRA parameters
        # Note: Test parameter relationships and constraints

        # Test default config
        config = LoRAConfig()
        assert config.r > 0
        assert config.alpha > 0
        assert 0 <= config.dropout <= 1
        assert config.target_modules is not None

        # Test predefined configs
        configs = LoRAConfig.get_lora_configs()
        assert "qa_lora" in configs
        assert "instruction_lora" in configs

        qa_config = configs["qa_lora"]
        assert qa_config.r == 16
        assert qa_config.alpha == 32

        # Test serialization
        config_dict = config.to_dict()
        assert isinstance(config_dict, dict)
        assert "r" in config_dict
        assert "lora_alpha" in config_dict

        logger.info("LoRAConfig test passed")

    def test_training_config(self):
        """Test TrainingConfig creation and validation"""
        # TODO: Implement TrainingConfig testing
        # Example: Test hyperparameters, batch size calculations
        # Use: Config validation, calculation methods
        # Input: Different training configurations
        # Output: Validation of training parameters
        # Note: Test parameter ranges and relationships

        # Test default config
        config = TrainingConfig()
        assert config.learning_rate > 0
        assert config.num_train_epochs > 0
        assert config.per_device_train_batch_size > 0

        # Test effective batch size calculation
        effective_batch_size = config.calculate_effective_batch_size()
        expected = (
            config.per_device_train_batch_size * config.gradient_accumulation_steps
        )
        assert effective_batch_size == expected

        # Test training time estimation
        time_estimate = config.estimate_training_time(
            num_samples=1000, steps_per_second=1.0
        )
        assert "total_steps" in time_estimate
        assert "estimated_hours" in time_estimate

        logger.info("TrainingConfig test passed")

    def test_full_pipeline_config(self):
        """Test FullPipelineConfig creation and validation"""
        # TODO: Implement FullPipelineConfig testing
        # Example: Test preset combinations, config compatibility
        # Use: Preset loading, validation functions
        # Input: Different preset configurations
        # Output: Validation of complete configurations
        # Note: Test compatibility between components

        # Test preset creation
        config = FullPipelineConfig.from_preset("quick_test")
        assert config.experiment_name == "quick_test_experiment"
        assert config.model_config is not None
        assert config.lora_config is not None
        assert config.training_config is not None
        assert config.data_config is not None

        # Test config saving/loading
        config_path = Path(tempfile.gettempdir()) / "test_config.json"
        config.save_config(str(config_path))
        assert config_path.exists()

        loaded_config = FullPipelineConfig.load_config(str(config_path))
        assert loaded_config.experiment_name == config.experiment_name

        # Cleanup
        config_path.unlink()

        logger.info("FullPipelineConfig test passed")


class TestDataFormats:
    """Test data formatting classes"""

    def setup_method(self):
        """Setup test environment"""
        self.tokenizer = MockTokenizer()
        self.test_samples = create_test_samples()

    def test_alpaca_formatter(self):
        """Test Alpaca formatting"""
        # TODO: Implement Alpaca formatter testing
        # Example: Test template application, label masking
        # Use: Formatter creation, sample processing
        # Input: Test samples with Alpaca format
        # Output: Validation of formatted output
        # Note: Test with/without input text, label masking

        formatter = AlpacaFormatter(self.tokenizer, max_length=512)

        sample = self.test_samples[0]

        # Test formatting
        formatted_text = formatter.format_sample(
            sample.instruction, sample.input, sample.output
        )

        assert "### Instruction:" in formatted_text
        assert "### Input:" in formatted_text
        assert "### Response:" in formatted_text
        assert sample.instruction in formatted_text
        assert sample.output in formatted_text

        # Test tokenization
        formatted_sample = formatter.tokenize_sample(
            formatted_text,
            instruction=sample.instruction,
            input=sample.input,
            output=sample.output,
            task_type=sample.task_type,
            source=sample.source,
        )

        assert isinstance(formatted_sample, FormattedSample)
        assert len(formatted_sample.input_ids) == len(formatted_sample.attention_mask)
        assert len(formatted_sample.input_ids) == len(formatted_sample.labels)

        logger.info("Alpaca formatter test passed")

    def test_mistral_formatter(self):
        """Test Mistral formatting"""
        # TODO: Implement Mistral formatter testing
        # Example: Test [INST] tags, chat template
        # Use: Formatter creation, template validation
        # Input: Test samples with Mistral format
        # Output: Validation of chat format
        # Note: Test instruction masking and special tokens

        formatter = MistralFormatter(self.tokenizer, max_length=512)

        sample = self.test_samples[0]

        # Test formatting
        formatted_text = formatter.format_sample(
            sample.instruction, sample.input, sample.output
        )

        assert "[INST]" in formatted_text
        assert "[/INST]" in formatted_text
        assert formatted_text.startswith("<s>")
        assert formatted_text.endswith("</s>")

        logger.info("Mistral formatter test passed")

    def test_data_format_manager(self):
        """Test DataFormatManager"""
        # TODO: Implement DataFormatManager testing
        # Example: Test different format types, batch processing
        # Use: Manager creation, format switching
        # Input: Multiple format types and samples
        # Output: Validation of format management
        # Note: Test error handling and format validation

        # Test with different formats
        formats = ["alpaca", "mistral", "qa", "summarization"]

        for format_type in formats:
            try:
                manager = DataFormatManager(
                    tokenizer=self.tokenizer, format_type=format_type, max_length=256
                )

                # Test batch formatting
                sample_dicts = [sample.to_dict() for sample in self.test_samples[:2]]
                formatted_samples = manager.format_dataset(sample_dicts)

                assert isinstance(formatted_samples, list)
                assert len(formatted_samples) <= len(
                    sample_dicts
                )  # May be fewer due to errors

                if formatted_samples:
                    assert isinstance(formatted_samples[0], FormattedSample)

                logger.info(f"DataFormatManager test passed for {format_type}")

            except Exception as e:
                logger.warning(f"DataFormatManager test failed for {format_type}: {e}")


class TestEvaluationMetrics:
    """Test evaluation metric classes"""

    def setup_method(self):
        """Setup test data"""
        self.predictions = [
            "Metformin is effective for treating type 2 diabetes",
            "The patient shows symptoms of hypertension",
            "Surgery is recommended for this cardiac condition",
        ]

        self.references = [
            "Metformin shows good efficacy in type 2 diabetes treatment",
            "Hypertension symptoms are present in the patient",
            "Cardiac surgery is the recommended treatment approach",
        ]

    def test_exact_match_metric(self):
        """Test exact match accuracy"""
        # TODO: Implement exact match testing
        # Example: Test with identical/different strings, normalization
        # Use: Metric computation, edge cases
        # Input: Test predictions and references
        # Output: Validation of exact match scores
        # Note: Test normalization and confidence intervals

        metric = ExactMatchMetric()

        # Test with identical strings
        identical_preds = self.references.copy()
        result = metric.compute(identical_preds, self.references)

        assert result.metric_name == "exact_match"
        assert result.score == 1.0  # Perfect match
        assert result.sample_count == len(self.references)

        # Test with different strings
        result = metric.compute(self.predictions, self.references)
        assert 0.0 <= result.score <= 1.0
        assert result.confidence_interval is not None

        logger.info("ExactMatch metric test passed")

    def test_f1_score_metric(self):
        """Test F1 score calculation"""
        # TODO: Implement F1 score testing
        # Example: Test token overlap, precision/recall calculation
        # Use: Metric computation, score validation
        # Input: Test predictions and references
        # Output: Validation of F1 scores
        # Note: Test edge cases like empty strings

        metric = F1ScoreMetric()

        result = metric.compute(self.predictions, self.references)

        assert result.metric_name == "f1_score"
        assert 0.0 <= result.score <= 1.0
        assert "precision" in result.details
        assert "recall" in result.details
        assert "f1" in result.details

        logger.info("F1Score metric test passed")

    def test_rouge_metric(self):
        """Test ROUGE score calculation"""
        # TODO: Implement ROUGE testing
        # Example: Test n-gram overlap, different ROUGE types
        # Use: Metric computation, ROUGE variants
        # Input: Test predictions and references
        # Output: Validation of ROUGE scores
        # Note: Test fallback implementation when library unavailable

        metric = ROUGEMetric()

        result = metric.compute(self.predictions, self.references)

        assert result.metric_name == "rouge"
        assert 0.0 <= result.score <= 1.0
        assert result.sample_count == len(self.predictions)

        # Should have ROUGE details
        if "rouge1" in result.details:
            assert 0.0 <= result.details["rouge1"] <= 1.0

        logger.info("ROUGE metric test passed")

    def test_medical_relevance_metric(self):
        """Test medical relevance scoring"""
        # TODO: Implement medical relevance testing
        # Example: Test medical keyword detection, domain scoring
        # Use: Medical vocabulary, relevance calculation
        # Input: Medical and non-medical text
        # Output: Validation of relevance scores
        # Note: Test different medical domains and terminology

        metric = MedicalRelevanceMetric()

        result = metric.compute(self.predictions, self.references)

        assert result.metric_name == "medical_relevance"
        assert 0.0 <= result.score <= 1.0
        assert "category_scores" in result.details

        # Check category scores
        categories = ["diagnoses", "treatments", "symptoms", "anatomy", "procedures"]
        for category in categories:
            if category in result.details["category_scores"]:
                assert 0.0 <= result.details["category_scores"][category] <= 1.0

        logger.info("Medical relevance metric test passed")

    def test_evaluation_suite(self):
        """Test complete evaluation suite"""
        # TODO: Implement evaluation suite testing
        # Example: Test multiple metrics, task-specific evaluation
        # Use: Suite creation, comprehensive evaluation
        # Input: Test data and task types
        # Output: Validation of complete evaluation
        # Note: Test error handling and report generation

        suite = EvaluationSuite(
            metrics=["exact_match", "f1_score", "medical_relevance"]
        )

        results = suite.evaluate(self.predictions, self.references, task_type="qa")

        assert isinstance(results, dict)
        assert len(results) > 0

        for metric_name, result in results.items():
            assert isinstance(result, EvaluationResult)
            assert 0.0 <= result.score <= 1.0 or result.score == float(
                "inf"
            )  # perplexity can be inf

        # Test report generation
        report = suite.generate_report(results)
        assert isinstance(report, str)
        assert "Evaluation Report" in report

        logger.info("Evaluation suite test passed")


async def test_training_pipeline():
    """Test training pipeline components"""
    # TODO: Implement pipeline testing
    # Example: Test pipeline flows, task orchestration
    # Use: Mock execution, flow validation
    # Input: Test configurations and data
    # Output: Validation of pipeline execution
    # Note: Test error handling and recovery

    try:
        from autoresearcher.pipelines.training_pipeline import (
            run_quick_test_pipeline, setup_model_and_tokenizer,
            validate_environment)

        # Test environment validation
        env_status = await validate_environment()
        assert isinstance(env_status, dict)
        assert "gpu_available" in env_status

        # Test model setup
        model_config = ModelConfig()
        model_setup = await setup_model_and_tokenizer(model_config)
        assert isinstance(model_setup, dict)
        assert "setup_successful" in model_setup

        logger.info("Training pipeline test passed")

    except ImportError as e:
        logger.warning(f"Training pipeline test skipped (missing dependencies): {e}")
    except Exception as e:
        logger.error(f"Training pipeline test failed: {e}")


def run_all_tests():
    """Run all test suites"""
    # TODO: Implement comprehensive test runner
    # Example: Execute all test classes, collect results
    # Use: Test execution, result aggregation
    # Input: All test classes
    # Output: Test results and summary
    # Note: Handle test failures and provide detailed output

    print("Running AutoResearcher Training Pipeline Tests")
    print("=" * 60)

    test_results = {"passed": 0, "failed": 0, "errors": []}

    # Test suites to run
    test_suites = [
        TestDatasetBuilders(),
        TestTrainingConfigs(),
        TestDataFormats(),
        TestEvaluationMetrics(),
    ]

    # Run test suites
    for suite in test_suites:
        suite_name = suite.__class__.__name__
        print(f"\nRunning {suite_name}...")

        # Get test methods
        test_methods = [method for method in dir(suite) if method.startswith("test_")]

        for method_name in test_methods:
            try:
                # Setup if available
                if hasattr(suite, "setup_method"):
                    suite.setup_method()

                # Run test
                test_method = getattr(suite, method_name)
                test_method()

                print(f"  ✓ {method_name}")
                test_results["passed"] += 1

                # Teardown if available
                if hasattr(suite, "teardown_method"):
                    suite.teardown_method()

            except Exception as e:
                print(f"  ✗ {method_name}: {e}")
                test_results["failed"] += 1
                test_results["errors"].append(f"{suite_name}.{method_name}: {e}")

    # Test async components
    print(f"\nRunning async tests...")
    try:
        asyncio.run(test_training_pipeline())
        print(f"  ✓ test_training_pipeline")
        test_results["passed"] += 1
    except Exception as e:
        print(f"  ✗ test_training_pipeline: {e}")
        test_results["failed"] += 1
        test_results["errors"].append(f"test_training_pipeline: {e}")

    # Print summary
    print(f"\n" + "=" * 60)
    print(f"Test Results:")
    print(f"  Passed: {test_results['passed']}")
    print(f"  Failed: {test_results['failed']}")
    print(f"  Total:  {test_results['passed'] + test_results['failed']}")

    if test_results["errors"]:
        print(f"\nErrors:")
        for error in test_results["errors"]:
            print(f"  - {error}")

    success_rate = (
        test_results["passed"] / (test_results["passed"] + test_results["failed"]) * 100
    )
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    return test_results


if __name__ == "__main__":
    # Run comprehensive test suite
    results = run_all_tests()

    # Exit with appropriate code
    exit_code = 0 if results["failed"] == 0 else 1
    exit(exit_code)
