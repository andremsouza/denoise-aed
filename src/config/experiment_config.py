"""Experiment configuration.

This module contains the dataclass-based configuration system for experiments.
"""

from dataclasses import dataclass, field, asdict
import os
from typing import Any
from urllib.parse import quote_plus

import dotenv
import optuna
import torch

from src.config.model_config import AudioModelConfig

dotenv.load_dotenv(override=True)

# Database connection parameters
POSTGRES_USER_OPTUNA = os.getenv("POSTGRES_USER_OPTUNA", "postgres")
POSTGRES_PASSWORD_OPTUNA = os.getenv("POSTGRES_PASSWORD_OPTUNA", "postgres")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "optuna")
# MLflow tracking parameters
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


@dataclass
class InfrastructureConfig:
    """Infrastructure configuration parameters.

    Args:
        use_mlflow: Whether to use MLflow for tracking experiments
    """

    use_mlflow: bool = True
    mlflow_tracking_uri: str = MLFLOW_TRACKING_URI
    mlflow_log_models: bool = False
    mlflow_log_datasets: bool = False
    mlflow_checkpoint: bool = True
    mlflow_checkpoint_monitor: str = "val_auroc"
    mlflow_checkpoint_mode: str = "max"
    mlflow_checkpoint_save_best_only: bool = True
    mlflow_checkpoint_save_weights_only: bool = False
    mlflow_checkpoint_save_freq: str = "epoch"
    mlflow_log_system_metrics: bool = True
    optuna_metric: str = "val_auroc"
    optuna_direction: str = "maximize"
    optuna_n_trials: int = 20
    optuna_n_jobs: int = 4
    optuna_timeout: int | None = None
    optuna_storage: str = (
        f"postgresql+psycopg2://{POSTGRES_USER_OPTUNA}:{quote_plus(POSTGRES_PASSWORD_OPTUNA)}@"
        f"{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
    )


@dataclass
class DataConfig:
    """Data configuration parameters.

    Args:
        data_directory: Directory containing raw audio data
        annotation_file: Path to the main annotation file
        train_annotation_file: Path to the training annotation file
        val_annotation_file: Path to the validation annotation file
        test_annotation_file: Path to the test annotation file
        sample_rate: Audio sample rate
        num_bands: Number of frequency bands
    """

    data_directory: str = "./data/aswine/audio/"
    annotation_file: str = "./data/aswine/meta/aswine_raw_full.csv"
    train_annotation_file: str = "./data/aswine/meta/train_aswine_raw_full.csv"
    val_annotation_file: str = "./data/aswine/meta/val_aswine_raw_full.csv"
    test_annotation_file: str = "./data/aswine/meta/test_aswine_raw_full.csv"
    sample_rate: int = 16000
    num_bands: int = 128
    train_mean: float | None = -0.0003
    train_std: float | None = 0.2069
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15


@dataclass
class TrainingConfig:
    """Model and training configuration parameters.

    Args:
        models_directory: Directory to save model checkpoints
        log_directory: Directory to save training logs
        batch_size: Training batch size
        chunk_size: Size of data chunks for processing
        n_mfccs: Number of MFCCs to extract
        use_pretrained: Whether to use pretrained models
        max_epochs: Maximum number of training epochs
        early_stopping_patience: Patience for early stopping
        num_workers: Number of workers for data loading
        device: Device to use for training (cuda or cpu)
        random_seed: Random seed for reproducibility
    """

    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    models_directory: str = "./models/"
    log_directory: str = "./logs/"
    use_pretrained: bool = False
    max_epochs: int = 1000
    early_stopping_patience: int = 4
    num_workers: int = min(8, (os.cpu_count() or 1) // 4)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    pin_memory: bool = False
    prefetch_factor: int | None = None


@dataclass
class AnnotationConfig:
    """Annotation-related configuration parameters.

    Args:
        num_classes: Number of classes for classification
        sample_seconds: Length of audio samples in seconds
        annotation_seconds: Length of annotations in seconds
    """

    num_classes: int = 7
    sample_seconds: float = 1.0
    annotation_seconds: int = 5


@dataclass
class EvaluationConfig:
    """Evaluation and utility configuration parameters.

    Args:
        pred_threshold: Threshold for making predictions
        skip_trained_models: Whether to skip already trained models
    """

    pred_threshold: float = 0.5
    skip_trained_models: bool = False


@dataclass
class ExperimentConfig(object):
    """Complete experiment configuration.

    This combines all configuration components into a single dataclass.

    Args:
        infrastructure: Infrastructure configuration
        data: Data configuration
        model: Model and training configuration
        annotation: Annotation configuration
        evaluation: Evaluation configuration
    """

    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: AudioModelConfig = field(default_factory=AudioModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    annotation: AnnotationConfig = field(default_factory=AnnotationConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def create_default_config(cls) -> "ExperimentConfig":
        """Create a default configuration instance."""
        return cls()

    def to_dict(self, flat: bool = False) -> dict:
        """Convert the configuration to a dictionary.

        Args:
            flat: If True, returns a flattened dictionary with keys like
                 'component_parameter'. If False, returns a nested dictionary
                 preserving the configuration structure.

        Returns:
            A dictionary representation of the configuration.
        """
        if not flat:
            # Return a nested dictionary preserving the structure
            return {
                "infrastructure": asdict(self.infrastructure),
                "data": asdict(self.data),
                "model": asdict(self.model),
                "training": asdict(self.training),
                "annotation": asdict(self.annotation),
                "evaluation": asdict(self.evaluation),
            }
        else:
            # Return a flattened dictionary
            flat_dict = {}

            # Process each configuration component
            for component_name, component in [
                ("infrastructure", self.infrastructure),
                ("data", self.data),
                ("model", self.model),
                ("training", self.training),
                ("annotation", self.annotation),
                ("evaluation", self.evaluation),
            ]:
                # Convert the component to a dictionary
                component_dict = asdict(component)
                # Add each parameter to the flat dictionary with the component as prefix
                for param_name, param_value in component_dict.items():
                    flat_key = f"{component_name}_{param_name}"
                    flat_dict[flat_key] = param_value

            return flat_dict

    @classmethod
    def from_trial(cls, trial: optuna.Trial, base_config=None):
        """Create a configuration from an Optuna trial.

        Args:
            trial: Optuna trial to suggest parameters from
            base_config: Base configuration to start with (optional)

        Returns:
            Configuration with parameters suggested by the trial
        """
        if base_config is None:
            config = cls.create_default_config()
        else:
            # Create a deep copy of the base config
            from copy import deepcopy

            config = deepcopy(base_config)

        # Suggest training parameters
        config.training.batch_size = trial.suggest_categorical(
            "batch_size", [16, 32, 64, 128]
        )
        config.training.learning_rate = trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        )
        config.training.weight_decay = trial.suggest_float(
            "weight_decay", 1e-6, 1e-2, log=True
        )

        # Model-specific parameters
        if isinstance(config.model, AudioModelConfig):
            config.model.learning_rate = config.training.learning_rate
            config.model.weight_decay = config.training.weight_decay

            # AST-specific parameters
            if hasattr(config.model, "fstride") and hasattr(config.model, "tstride"):
                config.model.fstride = trial.suggest_int("fstride", 10, 10)
                config.model.tstride = trial.suggest_int("tstride", 10, 10)

        return config
