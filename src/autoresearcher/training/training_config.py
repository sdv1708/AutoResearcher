"""
Training Configuration Classes for AutoResearcher Fine-tuning Pipeline

This module provides configuration classes for model training, LoRA/QLoRA settings,
and hyperparameter management.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union


@dataclass
class ModelConfig:
    """Configuration for different model types"""

    # Model identification
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    model_type: str = "mistral"  # "mistral", "biobert", "llama2"
    cache_dir: str = "./models/cache"

    # Model loading parameters
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = False

    # Tokenizer settings
    tokenizer_name: Optional[str] = None  # Use model_name if None
    max_sequence_length: int = 2048
    padding_side: str = "right"
    truncation_side: str = "right"

    # Model-specific parameters
    model_kwargs: Dict = field(default_factory=dict)

    def __post_init__(self):
        """Set model-specific defaults after initialization"""
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name

        # Set model-specific configurations
        if self.model_type == "mistral":
            self.model_kwargs.update(
                {"attn_implementation": "flash_attention_2", "use_cache": False}
            )
        elif self.model_type == "biobert":
            self.max_sequence_length = 512
            self.model_kwargs.update(
                {"hidden_dropout_prob": 0.1, "attention_probs_dropout_prob": 0.1}
            )
        elif self.model_type == "llama2":
            self.model_kwargs.update({"use_cache": False, "pad_token_id": 0})

    @classmethod
    def get_model_configs(cls) -> Dict[str, "ModelConfig"]:
        """Get predefined model configurations"""
        # TODO: Add more model configurations as needed
        # Example: Include BioMedLM, PubMedBERT, ClinicalBERT configurations
        # Use: Model-specific hyperparameters from literature
        # Input: None
        # Output: Dict[str, ModelConfig] with predefined configs
        # Note: Include both instruction-tuned and base models

        configs = {
            "mistral-7b": cls(
                model_name="mistralai/Mistral-7B-Instruct-v0.2",
                model_type="mistral",
                max_sequence_length=2048,
                load_in_8bit=True,
            ),
            "biobert": cls(
                model_name="dmis-lab/biobert-base-cased-v1.2",
                model_type="biobert",
                max_sequence_length=512,
                load_in_8bit=False,
                torch_dtype="float32",
            ),
            "llama2-7b": cls(
                model_name="meta-llama/Llama-2-7b-chat-hf",
                model_type="llama2",
                max_sequence_length=2048,
                load_in_8bit=True,
            ),
            "biomedlm": cls(
                model_name="stanford-crfm/BioMedLM",
                model_type="biomedlm",
                max_sequence_length=1024,
                load_in_8bit=True,
            ),
        }
        return configs

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "cache_dir": self.cache_dir,
            "load_in_8bit": self.load_in_8bit,
            "load_in_4bit": self.load_in_4bit,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "trust_remote_code": self.trust_remote_code,
            "tokenizer_name": self.tokenizer_name,
            "max_sequence_length": self.max_sequence_length,
            "padding_side": self.padding_side,
            "truncation_side": self.truncation_side,
            "model_kwargs": self.model_kwargs,
        }


@dataclass
class LoRAConfig:
    """Configuration for LoRA/QLoRA fine-tuning"""

    # LoRA parameters
    r: int = 16  # Rank of adaptation
    alpha: int = 32  # LoRA scaling parameter
    dropout: float = 0.1  # LoRA dropout
    bias: str = "none"  # Bias type: "none", "all", "lora_only"

    # Target modules for LoRA
    target_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None

    # QLoRA specific settings
    use_qlora: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # Task-specific settings
    task_type: str = "CAUSAL_LM"  # "CAUSAL_LM", "SEQ_CLS", "TOKEN_CLS"

    def __post_init__(self):
        """Set default target modules based on common architectures"""
        if self.target_modules is None:
            # Default LoRA targets for common architectures
            self.target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",  # Attention layers
                "gate_proj",
                "up_proj",
                "down_proj",  # MLP layers
            ]

    @classmethod
    def get_lora_configs(cls) -> Dict[str, "LoRAConfig"]:
        """Get predefined LoRA configurations for different use cases"""
        # TODO: Define task-specific LoRA configurations
        # Example: Different ranks and targets for QA vs summarization
        # Use: Literature recommendations, empirical results
        # Input: None
        # Output: Dict[str, LoRAConfig] with predefined configs
        # Note: Balance between parameter efficiency and performance

        configs = {
            "qa_lora": cls(
                r=16,
                alpha=32,
                dropout=0.1,
                target_modules=["q_proj", "v_proj", "o_proj"],
            ),
            "summarization_lora": cls(
                r=32,
                alpha=64,
                dropout=0.05,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
            ),
            "instruction_lora": cls(
                r=64,
                alpha=128,
                dropout=0.1,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                ],
                modules_to_save=["embed_tokens", "lm_head"],
            ),
            "efficient_lora": cls(
                r=8, alpha=16, dropout=0.1, target_modules=["q_proj", "v_proj"]
            ),
        }
        return configs

    def to_dict(self) -> Dict:
        """Convert to dictionary for PEFT configuration"""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "bias": self.bias,
            "target_modules": self.target_modules,
            "modules_to_save": self.modules_to_save,
            "task_type": self.task_type,
        }


@dataclass
class TrainingConfig:
    """Training hyperparameters and settings"""

    # Training hyperparameters
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4

    # Optimization settings
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Training behavior
    evaluation_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    logging_steps: int = 100

    # Early stopping and checkpointing
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"

    # Data processing
    max_seq_length: int = 2048
    data_collator_type: str = "default"  # "default", "completion_only"

    # Miscellaneous
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    dataloader_pin_memory: bool = True
    dataloader_num_workers: int = 4

    # Reporting and logging
    report_to: List[str] = field(default_factory=lambda: ["tensorboard"])
    run_name: Optional[str] = None
    output_dir: str = "./models/fine_tuned"
    logging_dir: Optional[str] = None

    def __post_init__(self):
        """Set dependent defaults after initialization"""
        if self.logging_dir is None:
            self.logging_dir = f"{self.output_dir}/logs"

        if self.run_name is None:
            from datetime import datetime

            self.run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @classmethod
    def get_training_configs(cls) -> Dict[str, "TrainingConfig"]:
        """Get predefined training configurations for different scenarios"""
        # TODO: Define scenario-specific training configurations
        # Example: Quick training, full training, evaluation-only configs
        # Use: Different batch sizes, learning rates for different hardware
        # Input: None
        # Output: Dict[str, TrainingConfig] with predefined configs
        # Note: Consider GPU memory constraints and training time

        configs = {
            "quick_test": cls(
                num_train_epochs=1,
                per_device_train_batch_size=2,
                eval_steps=50,
                save_steps=50,
                logging_steps=10,
            ),
            "full_training": cls(
                num_train_epochs=5,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=8,
                eval_steps=500,
                save_steps=500,
            ),
            "high_memory": cls(
                per_device_train_batch_size=8,
                gradient_accumulation_steps=2,
                dataloader_num_workers=8,
            ),
            "low_memory": cls(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=16,
                fp16=True,
                dataloader_num_workers=0,
            ),
        }
        return configs

    def calculate_effective_batch_size(self) -> int:
        """Calculate effective batch size considering gradient accumulation"""
        return self.per_device_train_batch_size * self.gradient_accumulation_steps

    def estimate_training_time(
        self, num_samples: int, steps_per_second: float = 1.0
    ) -> Dict[str, float]:
        """Estimate training time based on dataset size"""
        # TODO: Implement training time estimation
        # Example: Calculate based on batch size, epochs, and hardware speed
        # Use: Hardware benchmarks, empirical measurements
        # Input: num_samples (int), steps_per_second (float)
        # Output: Dict with time estimates in different units
        # Note: Include both training and evaluation time

        effective_batch_size = self.calculate_effective_batch_size()
        steps_per_epoch = num_samples // effective_batch_size
        total_steps = steps_per_epoch * self.num_train_epochs

        # Add evaluation steps
        eval_steps_total = (total_steps // self.eval_steps) * (
            num_samples // self.per_device_eval_batch_size
        )

        total_time_seconds = (total_steps + eval_steps_total) / steps_per_second

        return {
            "total_steps": total_steps,
            "steps_per_epoch": steps_per_epoch,
            "estimated_seconds": total_time_seconds,
            "estimated_minutes": total_time_seconds / 60,
            "estimated_hours": total_time_seconds / 3600,
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for TrainingArguments"""
        return {
            "learning_rate": self.learning_rate,
            "num_train_epochs": self.num_train_epochs,
            "per_device_train_batch_size": self.per_device_train_batch_size,
            "per_device_eval_batch_size": self.per_device_eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "optimizer": self.optimizer,
            "lr_scheduler_type": self.lr_scheduler_type,
            "warmup_ratio": self.warmup_ratio,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "evaluation_strategy": self.evaluation_strategy,
            "eval_steps": self.eval_steps,
            "save_strategy": self.save_strategy,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
            "save_total_limit": self.save_total_limit,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "seed": self.seed,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "dataloader_pin_memory": self.dataloader_pin_memory,
            "dataloader_num_workers": self.dataloader_num_workers,
            "report_to": self.report_to,
            "run_name": self.run_name,
            "output_dir": self.output_dir,
            "logging_dir": self.logging_dir,
        }


@dataclass
class DataConfig:
    """Configuration for dataset processing and loading"""

    # Dataset paths and settings
    dataset_name: str = "merged_medical_dataset"
    dataset_path: str = "./data/training"
    cache_dir: str = "./data/cache"

    # Data processing parameters
    max_samples_per_dataset: Optional[int] = None
    train_split_ratio: float = 0.8
    val_split_ratio: float = 0.1
    test_split_ratio: float = 0.1

    # Text processing
    max_input_length: int = 1024
    max_output_length: int = 512
    truncation_strategy: str = (
        "longest_first"  # "longest_first", "only_first", "only_second"
    )

    # Data filtering
    min_input_length: int = 10
    min_output_length: int = 5
    max_input_output_ratio: float = 10.0
    remove_duplicates: bool = True

    # Task-specific settings
    instruction_template: str = "default"  # "default", "alpaca", "vicuna", "custom"
    response_template: str = "### Response:\n"

    # Data loading
    streaming: bool = False
    num_proc: int = 4
    batch_size: int = 1000

    def validate_splits(self) -> bool:
        """Validate that split ratios sum to 1.0"""
        total = self.train_split_ratio + self.val_split_ratio + self.test_split_ratio
        return abs(total - 1.0) < 1e-6

    @classmethod
    def get_data_configs(cls) -> Dict[str, "DataConfig"]:
        """Get predefined data configurations for different scenarios"""
        # TODO: Define data configurations for different use cases
        # Example: Large dataset, small dataset, streaming, evaluation-only
        # Use: Different batch sizes, processing strategies
        # Input: None
        # Output: Dict[str, DataConfig] with predefined configs
        # Note: Consider memory constraints and processing time

        configs = {
            "small_dataset": cls(
                max_samples_per_dataset=1000, batch_size=100, num_proc=2
            ),
            "large_dataset": cls(streaming=True, batch_size=10000, num_proc=8),
            "evaluation_only": cls(
                train_split_ratio=0.0, val_split_ratio=0.5, test_split_ratio=0.5
            ),
            "qa_focused": cls(
                max_input_length=512, max_output_length=128, instruction_template="qa"
            ),
        }
        return configs

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "dataset_name": self.dataset_name,
            "dataset_path": self.dataset_path,
            "cache_dir": self.cache_dir,
            "max_samples_per_dataset": self.max_samples_per_dataset,
            "train_split_ratio": self.train_split_ratio,
            "val_split_ratio": self.val_split_ratio,
            "test_split_ratio": self.test_split_ratio,
            "max_input_length": self.max_input_length,
            "max_output_length": self.max_output_length,
            "truncation_strategy": self.truncation_strategy,
            "min_input_length": self.min_input_length,
            "min_output_length": self.min_output_length,
            "max_input_output_ratio": self.max_input_output_ratio,
            "remove_duplicates": self.remove_duplicates,
            "instruction_template": self.instruction_template,
            "response_template": self.response_template,
            "streaming": self.streaming,
            "num_proc": self.num_proc,
            "batch_size": self.batch_size,
        }


@dataclass
class FullPipelineConfig:
    """Complete configuration for the training pipeline"""

    model_config: ModelConfig
    lora_config: LoRAConfig
    training_config: TrainingConfig
    data_config: DataConfig

    # Pipeline settings
    experiment_name: str = "medical_finetuning"
    use_wandb: bool = False
    use_mlflow: bool = False
    save_merged_model: bool = True
    push_to_hub: bool = False
    hub_model_id: Optional[str] = None

    # Evaluation settings
    eval_datasets: List[str] = field(default_factory=lambda: ["pubmedqa", "bioasq"])
    eval_metrics: List[str] = field(
        default_factory=lambda: ["bleu", "rouge", "bertscore"]
    )

    def __post_init__(self):
        """Validate configuration compatibility"""
        # Ensure sequence lengths are compatible
        if self.data_config.max_input_length > self.model_config.max_sequence_length:
            self.data_config.max_input_length = (
                self.model_config.max_sequence_length - 50
            )

        if self.training_config.max_seq_length != self.model_config.max_sequence_length:
            self.training_config.max_seq_length = self.model_config.max_sequence_length

    @classmethod
    def from_preset(cls, preset_name: str) -> "FullPipelineConfig":
        """Create configuration from preset combinations"""
        # TODO: Implement preset configuration combinations
        # Example: "quick_test", "full_training", "production" presets
        # Use: Compatible combinations of model, lora, training, data configs
        # Input: preset_name (str)
        # Output: FullPipelineConfig with compatible settings
        # Note: Ensure all components work well together

        model_configs = ModelConfig.get_model_configs()
        lora_configs = LoRAConfig.get_lora_configs()
        training_configs = TrainingConfig.get_training_configs()
        data_configs = DataConfig.get_data_configs()

        presets = {
            "quick_test": {
                "model": "mistral-7b",
                "lora": "efficient_lora",
                "training": "quick_test",
                "data": "small_dataset",
            },
            "full_training": {
                "model": "mistral-7b",
                "lora": "instruction_lora",
                "training": "full_training",
                "data": "large_dataset",
            },
            "biobert_qa": {
                "model": "biobert",
                "lora": "qa_lora",
                "training": "full_training",
                "data": "qa_focused",
            },
        }

        if preset_name not in presets:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(presets.keys())}"
            )

        preset = presets[preset_name]

        return cls(
            model_config=model_configs[preset["model"]],
            lora_config=lora_configs[preset["lora"]],
            training_config=training_configs[preset["training"]],
            data_config=data_configs[preset["data"]],
            experiment_name=f"{preset_name}_experiment",
        )

    def save_config(self, path: str):
        """Save complete configuration to JSON file"""
        # TODO: Implement configuration serialization
        # Example: Convert all dataclasses to dict, save as JSON with metadata
        # Use: json.dump, dataclass serialization
        # Input: path (str)
        # Output: Saved JSON file with complete configuration
        # Note: Include version info and timestamp for reproducibility

        config_dict = {
            "experiment_name": self.experiment_name,
            "model_config": self.model_config.to_dict(),
            "lora_config": self.lora_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "data_config": self.data_config.to_dict(),
            "use_wandb": self.use_wandb,
            "use_mlflow": self.use_mlflow,
            "save_merged_model": self.save_merged_model,
            "push_to_hub": self.push_to_hub,
            "hub_model_id": self.hub_model_id,
            "eval_datasets": self.eval_datasets,
            "eval_metrics": self.eval_metrics,
            "created_at": str(pd.Timestamp.now()),
            "config_version": "1.0",
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load_config(cls, path: str) -> "FullPipelineConfig":
        """Load configuration from JSON file"""
        # TODO: Implement configuration deserialization
        # Example: Load JSON, recreate dataclass objects with validation
        # Use: json.load, dataclass reconstruction
        # Input: path (str)
        # Output: FullPipelineConfig object
        # Note: Handle version compatibility and missing fields

        with open(path, "r") as f:
            config_dict = json.load(f)

        # Reconstruct dataclass objects
        model_config = ModelConfig(**config_dict["model_config"])
        lora_config = LoRAConfig(**config_dict["lora_config"])
        training_config = TrainingConfig(**config_dict["training_config"])
        data_config = DataConfig(**config_dict["data_config"])

        return cls(
            model_config=model_config,
            lora_config=lora_config,
            training_config=training_config,
            data_config=data_config,
            experiment_name=config_dict.get("experiment_name", "loaded_experiment"),
            use_wandb=config_dict.get("use_wandb", False),
            use_mlflow=config_dict.get("use_mlflow", False),
            save_merged_model=config_dict.get("save_merged_model", True),
            push_to_hub=config_dict.get("push_to_hub", False),
            hub_model_id=config_dict.get("hub_model_id"),
            eval_datasets=config_dict.get("eval_datasets", []),
            eval_metrics=config_dict.get("eval_metrics", []),
        )


# Convenience functions for configuration management
def create_config_from_args(**kwargs) -> FullPipelineConfig:
    """Create configuration from command line arguments or kwargs"""
    # TODO: Implement argument parsing and config creation
    # Example: Use argparse or click to handle command line arguments
    # Use: Config validation, default value handling
    # Input: **kwargs with config parameters
    # Output: FullPipelineConfig object
    # Note: Support both file-based and argument-based configuration

    # Extract configuration components
    model_kwargs = {
        k.replace("model_", ""): v for k, v in kwargs.items() if k.startswith("model_")
    }
    lora_kwargs = {
        k.replace("lora_", ""): v for k, v in kwargs.items() if k.startswith("lora_")
    }
    training_kwargs = {
        k.replace("training_", ""): v
        for k, v in kwargs.items()
        if k.startswith("training_")
    }
    data_kwargs = {
        k.replace("data_", ""): v for k, v in kwargs.items() if k.startswith("data_")
    }

    # Create configuration objects
    model_config = ModelConfig(**model_kwargs)
    lora_config = LoRAConfig(**lora_kwargs)
    training_config = TrainingConfig(**training_kwargs)
    data_config = DataConfig(**data_kwargs)

    # Extract pipeline-level settings
    pipeline_kwargs = {
        k: v
        for k, v in kwargs.items()
        if not any(
            k.startswith(prefix) for prefix in ["model_", "lora_", "training_", "data_"]
        )
    }

    return FullPipelineConfig(
        model_config=model_config,
        lora_config=lora_config,
        training_config=training_config,
        data_config=data_config,
        **pipeline_kwargs,
    )


def validate_config_compatibility(config: FullPipelineConfig) -> List[str]:
    """Validate configuration for potential issues"""
    # TODO: Implement comprehensive configuration validation
    # Example: Check memory requirements, parameter compatibility
    # Use: Hardware detection, empirical memory formulas
    # Input: config (FullPipelineConfig)
    # Output: List[str] with validation warnings/errors
    # Note: Include performance and memory usage warnings

    warnings = []

    # Check sequence length compatibility
    if (
        config.data_config.max_input_length + config.data_config.max_output_length
        > config.model_config.max_sequence_length
    ):
        warnings.append("Input + output length exceeds model's maximum sequence length")

    # Check batch size vs memory
    effective_batch_size = config.training_config.calculate_effective_batch_size()
    if effective_batch_size > 32:
        warnings.append(
            f"Large effective batch size ({effective_batch_size}) may cause memory issues"
        )

    # Check LoRA configuration
    if config.lora_config.r > 64:
        warnings.append("High LoRA rank may reduce training efficiency")

    # Check split ratios
    if not config.data_config.validate_splits():
        warnings.append("Data split ratios do not sum to 1.0")

    return warnings


# Test function
def test_training_configs():
    """Test configuration classes and validation"""
    print("Testing training configuration classes...")

    # Test individual configs
    model_config = ModelConfig()
    lora_config = LoRAConfig()
    training_config = TrainingConfig()
    data_config = DataConfig()

    print(f"Model config created: {model_config.model_name}")
    print(f"LoRA rank: {lora_config.r}")
    print(f"Training epochs: {training_config.num_train_epochs}")
    print(f"Data config valid: {data_config.validate_splits()}")

    # Test full pipeline config
    full_config = FullPipelineConfig.from_preset("quick_test")
    print(f"Full config created: {full_config.experiment_name}")

    # Test validation
    warnings = validate_config_compatibility(full_config)
    print(f"Validation warnings: {len(warnings)}")

    print("Configuration testing completed successfully!")


if __name__ == "__main__":
    test_training_configs()
