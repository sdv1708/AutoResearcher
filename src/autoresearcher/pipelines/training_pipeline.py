"""
Prefect Training Pipeline for AutoResearcher Fine-tuning

This module provides orchestrated workflows for dataset preparation, model training,
evaluation, and deployment using Prefect for pipeline management.
"""

import json
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from prefect import flow, get_run_logger, task
from prefect.server.schemas.states import Completed, Failed
from prefect.tasks import task_input_hash

from ..training.data_formats import DataFormatManager
# Import our training components
from ..training.dataset_builder import (BioASQDatasetBuilder, DatasetMerger,
                                        InstructionDatasetBuilder,
                                        PubMedQADatasetBuilder)
from ..training.training_config import (DataConfig, FullPipelineConfig,
                                        LoRAConfig, ModelConfig,
                                        TrainingConfig)

logger = logging.getLogger(__name__)


@task(retries=3, retry_delay_seconds=60)
def validate_environment() -> Dict[str, Any]:
    """
    Validate training environment and dependencies

    Returns:
        Dictionary with environment status
    """
    # TODO: Implement comprehensive environment validation
    # Example: Check GPU availability, memory, disk space, dependencies
    # Use: torch.cuda, psutil, pkg_resources
    # Input: None
    # Output: Dict with environment status and recommendations
    # Note: Include GPU memory estimation and dependency versions

    logger = get_run_logger()
    status = {
        "gpu_available": False,
        "gpu_memory_gb": 0,
        "disk_space_gb": 0,
        "python_version": "",
        "dependencies_ok": True,
        "recommendations": [],
    }

    try:
        # Check GPU availability
        try:
            import torch

            status["gpu_available"] = torch.cuda.is_available()
            if status["gpu_available"]:
                status["gpu_memory_gb"] = (
                    torch.cuda.get_device_properties(0).total_memory / 1e9
                )
                logger.info(f"GPU available: {status['gpu_memory_gb']:.1f}GB")
        except ImportError:
            status["dependencies_ok"] = False
            status["recommendations"].append("Install PyTorch with CUDA support")

        # Check disk space
        try:
            import shutil

            disk_usage = shutil.disk_usage("./")
            status["disk_space_gb"] = disk_usage.free / 1e9
            logger.info(f"Available disk space: {status['disk_space_gb']:.1f}GB")

            if status["disk_space_gb"] < 50:
                status["recommendations"].append(
                    "Low disk space - need at least 50GB for training"
                )
        except Exception as e:
            logger.warning(f"Could not check disk space: {e}")

        # Check Python version
        import sys

        status["python_version"] = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Check key dependencies
        required_packages = ["transformers", "datasets", "peft", "accelerate"]
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                status["dependencies_ok"] = False
                status["recommendations"].append(f"Install {package}")

        logger.info("Environment validation completed")
        return status

    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        raise


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=24))
def prepare_pubmedqa_dataset(config: DataConfig, force_rebuild: bool = False) -> str:
    """
    Prepare PubMedQA dataset for training

    Args:
        config: Data configuration
        force_rebuild: Whether to rebuild existing dataset

    Returns:
        Path to prepared dataset
    """
    # TODO: Implement PubMedQA dataset preparation
    # Example: Download, process, and format PubMedQA dataset
    # Use: PubMedQADatasetBuilder, caching, validation
    # Input: config (DataConfig), force_rebuild (bool)
    # Output: str path to saved dataset
    # Note: Handle download failures and data validation

    logger = get_run_logger()
    logger.info("Preparing PubMedQA dataset")

    dataset_path = Path(config.dataset_path) / "pubmedqa"

    # Check if dataset already exists
    if dataset_path.exists() and not force_rebuild:
        logger.info(f"PubMedQA dataset already exists at {dataset_path}")
        return str(dataset_path)

    try:
        # Initialize builder
        builder = PubMedQADatasetBuilder(
            output_dir=config.dataset_path, cache_dir=config.cache_dir
        )

        # Build dataset
        saved_path = builder.build_dataset("pubmedqa")

        logger.info(f"PubMedQA dataset prepared at {saved_path}")
        return saved_path

    except Exception as e:
        logger.error(f"Failed to prepare PubMedQA dataset: {e}")
        raise


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=24))
def prepare_bioasq_dataset(config: DataConfig, force_rebuild: bool = False) -> str:
    """
    Prepare BioASQ dataset for training

    Args:
        config: Data configuration
        force_rebuild: Whether to rebuild existing dataset

    Returns:
        Path to prepared dataset
    """
    # TODO: Implement BioASQ dataset preparation
    # Example: Download, process, and format BioASQ dataset
    # Use: BioASQDatasetBuilder, manual download handling
    # Input: config (DataConfig), force_rebuild (bool)
    # Output: str path to saved dataset
    # Note: BioASQ may require manual download and special licensing

    logger = get_run_logger()
    logger.info("Preparing BioASQ dataset")

    dataset_path = Path(config.dataset_path) / "bioasq"

    if dataset_path.exists() and not force_rebuild:
        logger.info(f"BioASQ dataset already exists at {dataset_path}")
        return str(dataset_path)

    try:
        builder = BioASQDatasetBuilder(
            output_dir=config.dataset_path, cache_dir=config.cache_dir
        )

        saved_path = builder.build_dataset("bioasq")

        logger.info(f"BioASQ dataset prepared at {saved_path}")
        return saved_path

    except Exception as e:
        logger.warning(f"BioASQ dataset preparation failed: {e}")
        # Return empty path for BioASQ since it requires manual setup
        return ""


@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=12))
def prepare_instruction_dataset(
    config: DataConfig, faiss_index_path: str, force_rebuild: bool = False
) -> str:
    """
    Prepare instruction dataset from document chunks

    Args:
        config: Data configuration
        faiss_index_path: Path to FAISS index with documents
        force_rebuild: Whether to rebuild existing dataset

    Returns:
        Path to prepared dataset
    """
    # TODO: Implement instruction dataset preparation
    # Example: Load FAISS index, create instruction tasks from documents
    # Use: InstructionDatasetBuilder, FAISS integration
    # Input: config (DataConfig), faiss_index_path (str), force_rebuild (bool)
    # Output: str path to saved dataset
    # Note: Handle missing FAISS index and ensure quality instruction generation

    logger = get_run_logger()
    logger.info("Preparing instruction dataset from documents")

    dataset_path = Path(config.dataset_path) / "instruction_dataset"

    if dataset_path.exists() and not force_rebuild:
        logger.info(f"Instruction dataset already exists at {dataset_path}")
        return str(dataset_path)

    try:
        # Check if FAISS index exists
        if not Path(faiss_index_path).exists():
            logger.warning(f"FAISS index not found at {faiss_index_path}")
            return ""

        builder = InstructionDatasetBuilder(
            faiss_index_path=faiss_index_path,
            output_dir=config.dataset_path,
            cache_dir=config.cache_dir,
        )

        saved_path = builder.build_dataset("instruction_dataset")

        logger.info(f"Instruction dataset prepared at {saved_path}")
        return saved_path

    except Exception as e:
        logger.error(f"Failed to prepare instruction dataset: {e}")
        raise


@task
def merge_datasets(
    dataset_paths: List[str], config: DataConfig, weights: Optional[List[float]] = None
) -> str:
    """
    Merge multiple datasets with optional weighting

    Args:
        dataset_paths: List of paths to datasets
        config: Data configuration
        weights: Optional weights for each dataset

    Returns:
        Path to merged dataset
    """
    # TODO: Implement dataset merging with deduplication
    # Example: Load datasets, apply weights, remove duplicates, shuffle
    # Use: DatasetMerger, hash-based deduplication
    # Input: dataset_paths (List[str]), config (DataConfig), weights (Optional[List[float]])
    # Output: str path to merged dataset
    # Note: Handle memory efficiently for large datasets

    logger = get_run_logger()
    logger.info(f"Merging {len(dataset_paths)} datasets")

    # Filter out empty paths
    valid_paths = [path for path in dataset_paths if path and Path(path).exists()]

    if not valid_paths:
        raise ValueError("No valid dataset paths provided for merging")

    try:
        merger = DatasetMerger(output_dir=f"{config.dataset_path}/merged")
        merged_path = merger.merge_datasets(
            dataset_paths=valid_paths, weights=weights, output_name=config.dataset_name
        )

        logger.info(f"Datasets merged at {merged_path}")
        return merged_path

    except Exception as e:
        logger.error(f"Failed to merge datasets: {e}")
        raise


@task
def validate_dataset(dataset_path: str, config: DataConfig) -> Dict[str, Any]:
    """
    Validate prepared dataset for training

    Args:
        dataset_path: Path to dataset
        config: Data configuration

    Returns:
        Validation results
    """
    # TODO: Implement comprehensive dataset validation
    # Example: Check splits, sample quality, token distributions
    # Use: datasets library, statistical validation
    # Input: dataset_path (str), config (DataConfig)
    # Output: Dict with validation results and statistics
    # Note: Include recommendations for improving dataset quality

    logger = get_run_logger()
    logger.info(f"Validating dataset at {dataset_path}")

    try:
        from datasets import DatasetDict

        # Load dataset
        dataset = DatasetDict.load_from_disk(dataset_path)

        validation_results = {
            "dataset_path": dataset_path,
            "splits": list(dataset.keys()),
            "total_samples": sum(len(dataset[split]) for split in dataset.keys()),
            "split_sizes": {split: len(dataset[split]) for split in dataset.keys()},
            "validation_passed": True,
            "issues": [],
            "recommendations": [],
        }

        # Check split sizes
        for split in dataset.keys():
            if len(dataset[split]) == 0:
                validation_results["issues"].append(f"Empty {split} split")
                validation_results["validation_passed"] = False

        # Check sample structure
        if "train" in dataset and len(dataset["train"]) > 0:
            sample = dataset["train"][0]
            required_fields = ["instruction", "input", "output"]
            missing_fields = [field for field in required_fields if field not in sample]

            if missing_fields:
                validation_results["issues"].append(f"Missing fields: {missing_fields}")
                validation_results["validation_passed"] = False

        # Check for minimum samples
        min_samples = 100  # Minimum viable dataset size
        if validation_results["total_samples"] < min_samples:
            validation_results["recommendations"].append(
                f"Dataset has only {validation_results['total_samples']} samples, consider adding more data"
            )

        logger.info(
            f"Dataset validation completed: {validation_results['validation_passed']}"
        )
        return validation_results

    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        return {
            "dataset_path": dataset_path,
            "validation_passed": False,
            "error": str(e),
        }


@task
def setup_model_and_tokenizer(model_config: ModelConfig) -> Dict[str, Any]:
    """
    Setup model and tokenizer for training

    Args:
        model_config: Model configuration

    Returns:
        Model and tokenizer setup status
    """
    # TODO: Implement model and tokenizer setup
    # Example: Load model with quantization, prepare tokenizer
    # Use: transformers.AutoModel, AutoTokenizer, BitsAndBytesConfig
    # Input: model_config (ModelConfig)
    # Output: Dict with setup status and model info
    # Note: Handle different model types and quantization settings

    logger = get_run_logger()
    logger.info(f"Setting up model: {model_config.model_name}")

    try:
        # This is a validation step - actual model loading happens in training
        setup_status = {
            "model_name": model_config.model_name,
            "model_type": model_config.model_type,
            "quantization": (
                "8bit"
                if model_config.load_in_8bit
                else ("4bit" if model_config.load_in_4bit else "none")
            ),
            "max_length": model_config.max_sequence_length,
            "setup_successful": True,
            "estimated_memory_gb": 0,
        }

        # Estimate memory requirements
        model_size_map = {"7b": 14, "13b": 26, "70b": 140}  # 7B model ~14GB in fp16

        model_size = "7b"  # Default
        for size in model_size_map:
            if size in model_config.model_name.lower():
                model_size = size
                break

        base_memory = model_size_map[model_size]

        if model_config.load_in_8bit:
            setup_status["estimated_memory_gb"] = base_memory * 0.5
        elif model_config.load_in_4bit:
            setup_status["estimated_memory_gb"] = base_memory * 0.25
        else:
            setup_status["estimated_memory_gb"] = base_memory

        logger.info(
            f"Model setup completed: {setup_status['estimated_memory_gb']:.1f}GB estimated"
        )
        return setup_status

    except Exception as e:
        logger.error(f"Model setup failed: {e}")
        return {
            "model_name": model_config.model_name,
            "setup_successful": False,
            "error": str(e),
        }


@task
def run_training(
    dataset_path: str,
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    training_config: TrainingConfig,
    data_config: DataConfig,
) -> Dict[str, Any]:
    """
    Run model fine-tuning with LoRA/QLoRA

    Args:
        dataset_path: Path to training dataset
        model_config: Model configuration
        lora_config: LoRA configuration
        training_config: Training hyperparameters
        data_config: Data processing configuration

    Returns:
        Training results and metrics
    """
    # TODO: Implement actual model training
    # Example: Load model, apply LoRA, run training loop
    # Use: transformers.Trainer, peft.LoraConfig, accelerate
    # Input: All configuration objects
    # Output: Dict with training metrics, model path, checkpoints
    # Note: Handle GPU memory management, checkpointing, early stopping

    logger = get_run_logger()
    logger.info("Starting model training")

    try:
        # This is a placeholder implementation
        # In production, this would use transformers.Trainer

        training_results = {
            "training_started": datetime.now().isoformat(),
            "model_name": model_config.model_name,
            "dataset_path": dataset_path,
            "output_dir": training_config.output_dir,
            "epochs": training_config.num_train_epochs,
            "learning_rate": training_config.learning_rate,
            "lora_rank": lora_config.r,
            "training_successful": True,
            "final_metrics": {
                "train_loss": 0.5,  # Placeholder
                "eval_loss": 0.6,  # Placeholder
                "eval_accuracy": 0.75,  # Placeholder
            },
            "model_save_path": f"{training_config.output_dir}/final_model",
            "checkpoints": [
                f"{training_config.output_dir}/checkpoint-500",
                f"{training_config.output_dir}/checkpoint-1000",
            ],
        }

        # Simulate training time
        import time

        time.sleep(2)  # Placeholder for actual training

        training_results["training_completed"] = datetime.now().isoformat()

        logger.info("Training completed successfully")
        return training_results

    except Exception as e:
        logger.error(f"Training failed: {e}")
        return {
            "training_successful": False,
            "error": str(e),
            "model_name": model_config.model_name,
        }


@task
def evaluate_model(
    model_path: str, eval_datasets: List[str], eval_metrics: List[str]
) -> Dict[str, Any]:
    """
    Evaluate trained model on benchmark datasets

    Args:
        model_path: Path to trained model
        eval_datasets: List of evaluation datasets
        eval_metrics: List of metrics to compute

    Returns:
        Evaluation results
    """
    # TODO: Implement model evaluation on medical benchmarks
    # Example: Load model, run inference on test sets, compute metrics
    # Use: evaluate library, custom medical metrics
    # Input: model_path (str), eval_datasets (List[str]), eval_metrics (List[str])
    # Output: Dict with evaluation results per dataset and metric
    # Note: Include confidence intervals and statistical significance tests

    logger = get_run_logger()
    logger.info(f"Evaluating model at {model_path}")

    try:
        # Placeholder evaluation results
        evaluation_results = {
            "model_path": model_path,
            "evaluation_completed": datetime.now().isoformat(),
            "datasets_evaluated": eval_datasets,
            "metrics_computed": eval_metrics,
            "results": {},
        }

        # Simulate evaluation for each dataset
        for dataset in eval_datasets:
            dataset_results = {}
            for metric in eval_metrics:
                # Placeholder metric values
                if metric == "bleu":
                    dataset_results[metric] = 0.35
                elif metric == "rouge":
                    dataset_results[metric] = {
                        "rouge1": 0.45,
                        "rouge2": 0.25,
                        "rougeL": 0.40,
                    }
                elif metric == "bertscore":
                    dataset_results[metric] = {
                        "precision": 0.72,
                        "recall": 0.68,
                        "f1": 0.70,
                    }
                elif metric == "exact_match":
                    dataset_results[metric] = 0.62
                else:
                    dataset_results[metric] = 0.70

            evaluation_results["results"][dataset] = dataset_results

        logger.info("Model evaluation completed")
        return evaluation_results

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        return {
            "model_path": model_path,
            "evaluation_successful": False,
            "error": str(e),
        }


@task
def deploy_model(model_path: str, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deploy trained model to model registry or serving platform

    Args:
        model_path: Path to trained model
        deployment_config: Deployment configuration

    Returns:
        Deployment results
    """
    # TODO: Implement model deployment
    # Example: Upload to HuggingFace Hub, save to model registry
    # Use: huggingface_hub, mlflow, custom deployment
    # Input: model_path (str), deployment_config (Dict)
    # Output: Dict with deployment status and model URLs
    # Note: Handle authentication and versioning

    logger = get_run_logger()
    logger.info(f"Deploying model from {model_path}")

    try:
        deployment_results = {
            "model_path": model_path,
            "deployment_started": datetime.now().isoformat(),
            "deployment_successful": True,
            "model_registry_url": "https://huggingface.co/autoresearcher/medical-mistral-7b",
            "model_version": "v1.0.0",
            "deployment_type": deployment_config.get("type", "registry"),
        }

        # Simulate deployment
        import time

        time.sleep(1)

        deployment_results["deployment_completed"] = datetime.now().isoformat()

        logger.info("Model deployment completed")
        return deployment_results

    except Exception as e:
        logger.error(f"Model deployment failed: {e}")
        return {
            "model_path": model_path,
            "deployment_successful": False,
            "error": str(e),
        }


@flow(name="prepare-datasets")
def prepare_datasets_flow(
    config: FullPipelineConfig,
    faiss_index_path: str = "./data/faiss_index",
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """
    Prepare all datasets for training

    Args:
        config: Full pipeline configuration
        faiss_index_path: Path to FAISS index
        force_rebuild: Whether to rebuild existing datasets

    Returns:
        Dataset preparation results
    """
    logger = get_run_logger()
    logger.info("Starting dataset preparation flow")

    # Validate environment
    env_status = validate_environment()

    # Prepare individual datasets in parallel
    pubmedqa_future = prepare_pubmedqa_dataset.submit(config.data_config, force_rebuild)
    bioasq_future = prepare_bioasq_dataset.submit(config.data_config, force_rebuild)
    instruction_future = prepare_instruction_dataset.submit(
        config.data_config, faiss_index_path, force_rebuild
    )

    # Wait for completion
    pubmedqa_path = pubmedqa_future.result()
    bioasq_path = bioasq_future.result()
    instruction_path = instruction_future.result()

    # Collect valid dataset paths
    dataset_paths = [
        path for path in [pubmedqa_path, bioasq_path, instruction_path] if path
    ]

    if not dataset_paths:
        raise ValueError("No datasets were successfully prepared")

    # Merge datasets
    merged_path = merge_datasets(dataset_paths, config.data_config)

    # Validate merged dataset
    validation_results = validate_dataset(merged_path, config.data_config)

    return {
        "environment_status": env_status,
        "individual_datasets": {
            "pubmedqa": pubmedqa_path,
            "bioasq": bioasq_path,
            "instruction": instruction_path,
        },
        "merged_dataset_path": merged_path,
        "validation_results": validation_results,
        "flow_completed": datetime.now().isoformat(),
    }


@flow(name="training-flow")
def training_flow(config: FullPipelineConfig, dataset_path: str) -> Dict[str, Any]:
    """
    Model training flow

    Args:
        config: Full pipeline configuration
        dataset_path: Path to prepared dataset

    Returns:
        Training results
    """
    logger = get_run_logger()
    logger.info("Starting training flow")

    # Setup model and tokenizer
    model_setup = setup_model_and_tokenizer(config.model_config)

    if not model_setup["setup_successful"]:
        raise ValueError(f"Model setup failed: {model_setup.get('error')}")

    # Run training
    training_results = run_training(
        dataset_path=dataset_path,
        model_config=config.model_config,
        lora_config=config.lora_config,
        training_config=config.training_config,
        data_config=config.data_config,
    )

    return {
        "model_setup": model_setup,
        "training_results": training_results,
        "flow_completed": datetime.now().isoformat(),
    }


@flow(name="evaluation-flow")
def evaluation_flow(
    model_path: str, eval_datasets: List[str], eval_metrics: List[str]
) -> Dict[str, Any]:
    """
    Model evaluation flow

    Args:
        model_path: Path to trained model
        eval_datasets: List of evaluation datasets
        eval_metrics: List of metrics to compute

    Returns:
        Evaluation results
    """
    logger = get_run_logger()
    logger.info("Starting evaluation flow")

    # Run evaluation
    evaluation_results = evaluate_model(model_path, eval_datasets, eval_metrics)

    return {
        "evaluation_results": evaluation_results,
        "flow_completed": datetime.now().isoformat(),
    }


@flow(name="deployment-flow")
def deployment_flow(
    model_path: str, deployment_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Model deployment flow

    Args:
        model_path: Path to trained model
        deployment_config: Deployment configuration

    Returns:
        Deployment results
    """
    logger = get_run_logger()
    logger.info("Starting deployment flow")

    # Deploy model
    deployment_results = deploy_model(model_path, deployment_config)

    return {
        "deployment_results": deployment_results,
        "flow_completed": datetime.now().isoformat(),
    }


@flow(name="full-training-pipeline")
def full_pipeline_flow(
    config: FullPipelineConfig,
    faiss_index_path: str = "./data/faiss_index",
    force_rebuild_datasets: bool = False,
    skip_evaluation: bool = False,
    skip_deployment: bool = False,
) -> Dict[str, Any]:
    """
    Complete end-to-end training pipeline

    Args:
        config: Full pipeline configuration
        faiss_index_path: Path to FAISS index
        force_rebuild_datasets: Whether to rebuild datasets
        skip_evaluation: Whether to skip evaluation
        skip_deployment: Whether to skip deployment

    Returns:
        Complete pipeline results
    """
    logger = get_run_logger()
    logger.info("Starting full training pipeline")

    pipeline_start = datetime.now()

    try:
        # Step 1: Prepare datasets
        dataset_results = prepare_datasets_flow(
            config=config,
            faiss_index_path=faiss_index_path,
            force_rebuild=force_rebuild_datasets,
        )

        if not dataset_results["validation_results"]["validation_passed"]:
            raise ValueError("Dataset validation failed")

        # Step 2: Train model
        training_results = training_flow(
            config=config, dataset_path=dataset_results["merged_dataset_path"]
        )

        if not training_results["training_results"]["training_successful"]:
            raise ValueError("Training failed")

        model_path = training_results["training_results"]["model_save_path"]

        # Step 3: Evaluate model (optional)
        evaluation_results = None
        if not skip_evaluation:
            evaluation_results = evaluation_flow(
                model_path=model_path,
                eval_datasets=config.eval_datasets,
                eval_metrics=config.eval_metrics,
            )

        # Step 4: Deploy model (optional)
        deployment_results = None
        if not skip_deployment:
            deployment_config = {
                "type": "registry",
                "push_to_hub": config.push_to_hub,
                "hub_model_id": config.hub_model_id,
            }
            deployment_results = deployment_flow(
                model_path=model_path, deployment_config=deployment_config
            )

        pipeline_end = datetime.now()
        pipeline_duration = (pipeline_end - pipeline_start).total_seconds()

        return {
            "pipeline_successful": True,
            "pipeline_duration_seconds": pipeline_duration,
            "experiment_name": config.experiment_name,
            "dataset_results": dataset_results,
            "training_results": training_results,
            "evaluation_results": evaluation_results,
            "deployment_results": deployment_results,
            "model_path": model_path,
            "completed_at": pipeline_end.isoformat(),
        }

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        return {
            "pipeline_successful": False,
            "error": str(e),
            "experiment_name": config.experiment_name,
            "failed_at": datetime.now().isoformat(),
        }


# Convenience functions for running pipelines
def run_quick_test_pipeline(
    faiss_index_path: str = "./data/faiss_index",
) -> Dict[str, Any]:
    """Run a quick test of the training pipeline"""
    # TODO: Implement quick test pipeline execution
    # Example: Use minimal configuration, small dataset, fast training
    # Use: FullPipelineConfig.from_preset, minimal resources
    # Input: faiss_index_path (str)
    # Output: Dict with test results
    # Note: Should complete in under 30 minutes for testing

    config = FullPipelineConfig.from_preset("quick_test")

    return full_pipeline_flow(
        config=config,
        faiss_index_path=faiss_index_path,
        force_rebuild_datasets=False,
        skip_evaluation=True,
        skip_deployment=True,
    )


def run_full_training_pipeline(
    preset: str = "full_training",
    experiment_name: str = None,
    faiss_index_path: str = "./data/faiss_index",
) -> Dict[str, Any]:
    """Run the complete training pipeline"""
    # TODO: Implement full training pipeline execution
    # Example: Use production configuration, all datasets, full evaluation
    # Use: FullPipelineConfig.from_preset, complete workflow
    # Input: preset (str), experiment_name (str), faiss_index_path (str)
    # Output: Dict with complete pipeline results
    # Note: May take several hours to complete

    config = FullPipelineConfig.from_preset(preset)

    if experiment_name:
        config.experiment_name = experiment_name

    return full_pipeline_flow(
        config=config,
        faiss_index_path=faiss_index_path,
        force_rebuild_datasets=False,
        skip_evaluation=False,
        skip_deployment=False,
    )


if __name__ == "__main__":
    # Test the pipeline with a quick test
    print("Running quick test pipeline...")
    results = run_quick_test_pipeline()

    if results["pipeline_successful"]:
        print("Pipeline test completed successfully!")
        print(f"Duration: {results['pipeline_duration_seconds']:.1f} seconds")
    else:
        print(f"Pipeline test failed: {results['error']}")
