#!/usr/bin/env python
# filepath: /nfs/home/andre/intrinsic-dimension/cli.py
"""
Command-line interface for running intrinsic dimension experiments.
"""

import argparse
from datetime import datetime

import torch

from src.architectures import (
    PLAST,
    PLCnn14,
    PLResNet38,
    PLMobileNetV1,
    PLMobileNetV2,
    PLDaiNet19,
    PLLeeNet24,
)
from src.experiments.base import ExperimentRunner
from src.experiments.optuna_runner import OptunaExperimentRunner
from src.config.experiment_config import ExperimentConfig
from src.factories.model_config_factory import create_model_config

MODELS = {
    "ast": PLAST,
    "cnn14": PLCnn14,
    "dainet19": PLDaiNet19,
    "leenet24": PLLeeNet24,
    "mobilenetv1": PLMobileNetV1,
    "mobilenetv2": PLMobileNetV2,
    "resnet38": PLResNet38,
}


def main() -> None:
    """Parse command-line arguments and run the experiment."""
    # Create experiment configuration
    experiment_config = ExperimentConfig.create_default_config()

    # Parse command-line arguments
    args = parse_args(experiment_config=experiment_config)

    # Update infrastructure config
    experiment_config.infrastructure.use_mlflow = args.use_mlflow

    # Update data config
    experiment_config.data.data_directory = args.data_dir
    experiment_config.data.annotation_file = args.annotation_file
    experiment_config.data.sample_rate = args.sample_rate
    experiment_config.data.num_bands = args.num_bands

    # Update training config
    experiment_config.training.batch_size = args.batch_size
    experiment_config.training.models_directory = args.models_dir
    experiment_config.training.log_directory = args.logs_dir
    experiment_config.training.max_epochs = args.max_epochs
    experiment_config.training.early_stopping_patience = args.patience
    experiment_config.training.num_workers = args.num_workers
    experiment_config.training.device = args.device
    experiment_config.training.random_seed = args.seed
    experiment_config.training.use_pretrained = args.use_pretrained

    # Update annotation config
    experiment_config.annotation.num_classes = args.num_classes
    experiment_config.annotation.sample_seconds = args.sample_seconds

    # Update evaluation config
    experiment_config.evaluation.skip_trained_models = args.skip_trained
    experiment_config.evaluation.pred_threshold = args.pred_threshold

    # Create model config based on model type
    model_config = create_model_config(args.model_type, num_classes=args.num_classes)

    # Update model config with common settings
    model_config.learning_rate = args.lr
    model_config.weight_decay = args.weight_decay

    # Update AST-specific settings if applicable
    if args.model_type == "ast":
        model_config.imagenet_pretrain = args.ast_imagenet_pretrain
        model_config.audioset_pretrain = args.ast_audioset_pretrain
        model_config.model_size = args.ast_model_size

    # Assign model config to experiment config
    experiment_config.model = model_config

    # Set high precision for float32 matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Run the experiment
    experiment_prefix: str = "intrinsic_dimension"
    print(f"[{datetime.now()}]: Starting experiment with prefix: {experiment_prefix}")
    print(f"[{datetime.now()}]: Using model type: {args.model_type}")
    print(f"[{datetime.now()}]: Arguments: {args}")

    try:
        # Check if optimization is enabled
        if args.optimize or args.optimize_only:
            # Create OptunaExperimentRunner
            experiment = OptunaExperimentRunner(
                experiment_config=experiment_config,
                experiment_prefix=experiment_prefix,
                optimization_metric=args.optimize_metric,
                direction=args.optimize_direction,
                n_trials=args.n_trials,
                n_jobs=args.n_jobs,
                timeout=args.timeout,
                study_name=args.study_name,
                storage=args.storage,
                verbose=True,
            )

            # Run optimization and then final experiment with best parameters
            if args.optimize_only:
                # Just run optimization
                experiment.run_optimization(model_class=MODELS[args.model_type])
            else:
                # Run optimization and then final experiment
                experiment.run_with_best_params(model_class=MODELS[args.model_type])
        else:
            # Create regular ExperimentRunner
            experiment = ExperimentRunner(
                experiment_config=experiment_config,
                experiment_prefix=experiment_prefix,
                verbose=True,
            )
            experiment.run_experiment(model_class=MODELS[args.model_type])
    except KeyError:
        print(f"[{datetime.now()}]: Model type {args.model_type} not supported yet.")
        return

    print(f"[{datetime.now()}]: Experiment complete")


def parse_args(experiment_config: ExperimentConfig) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        experiment_config: Default configuration to use for argument defaults

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run intrinsic dimension experiments")

    # Infrastructure arguments
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        default=experiment_config.infrastructure.use_mlflow,
        help="Use MLflow for experiment tracking",
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default=experiment_config.data.data_directory,
        help="Directory containing audio data",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=experiment_config.data.annotation_file,
        help="Path to annotation file",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=experiment_config.data.sample_rate,
        help="Audio sample rate",
    )
    parser.add_argument(
        "--num-bands",
        type=int,
        default=experiment_config.data.num_bands,
        help="Number of frequency bands",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=experiment_config.training.batch_size,
        help="Training batch size",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=experiment_config.training.models_directory,
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--logs-dir",
        type=str,
        default=experiment_config.training.log_directory,
        help="Directory to save training logs",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=experiment_config.training.max_epochs,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=experiment_config.training.early_stopping_patience,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=experiment_config.training.num_workers,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=experiment_config.training.device,
        help="Device to use for training (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=experiment_config.training.random_seed,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--use-pretrained",
        "-p",
        action="store_true",
        default=experiment_config.training.use_pretrained,
        help="Use pretrained model checkpoints if available",
    )

    # Annotation arguments
    parser.add_argument(
        "--num-classes",
        type=int,
        default=experiment_config.annotation.num_classes,
        help="Number of classes for classification",
    )
    parser.add_argument(
        "--sample-seconds",
        type=float,
        default=experiment_config.annotation.sample_seconds,
        help="Length of audio samples in seconds",
    )

    # Evaluation arguments
    parser.add_argument(
        "--skip-trained",
        action="store_true",
        default=experiment_config.evaluation.skip_trained_models,
        help="Skip already trained models",
    )
    parser.add_argument(
        "--pred-threshold",
        type=float,
        default=experiment_config.evaluation.pred_threshold,
        help="Threshold for making predictions",
    )

    # Model arguments
    parser.add_argument(
        "--model-type",
        "-m",
        type=str,
        required=True,
        choices=[
            "ast",
            "cnn14",
            "dainet19",
            "leenet24",
            "mobilenetv1",
            "mobilenetv2",
            "resnet38",
        ],
        help="Type of model architecture to use",
    )

    # Model-specific arguments
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for optimizer",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay for optimizer",
    )

    # AST-specific arguments
    parser.add_argument(
        "--ast-imagenet-pretrain",
        action="store_true",
        help="Use ImageNet pretrained weights for AST",
    )
    parser.add_argument(
        "--no-ast-imagenet-pretrain",
        action="store_false",
        dest="ast_imagenet_pretrain",
        help="Don't use ImageNet pretrained weights for AST",
    )
    parser.add_argument(
        "--ast-audioset-pretrain",
        action="store_true",
        help="Use AudioSet pretrained weights for AST",
    )
    parser.add_argument(
        "--no-ast-audioset-pretrain",
        action="store_false",
        dest="ast_audioset_pretrain",
        help="Don't use AudioSet pretrained weights for AST",
    )
    parser.add_argument(
        "--ast-model-size",
        type=str,
        default="base384",
        choices=["base384", "small384"],
        help="Model size for AST",
    )

    # Optuna arguments
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization with Optuna",
    )
    parser.add_argument(
        "--optimize-only",
        action="store_true",
        help="Run only optimization without final experiment",
    )
    parser.add_argument(
        "--optimize-metric",
        type=str,
        default=experiment_config.infrastructure.optuna_metric,
        choices=[
            "val_auroc",
            "val_loss",
            "val_accuracy",
            "val_f1",
            "val_precision",
            "val_recall",
        ],
        help="Metric to optimize ('val_auroc', 'val_loss', etc.)",
    )
    parser.add_argument(
        "--optimize-direction",
        type=str,
        default=experiment_config.infrastructure.optuna_direction,
        choices=["maximize", "minimize"],
        help="Direction of optimization",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=experiment_config.infrastructure.optuna_n_trials,
        help="Number of optimization trials to run",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=experiment_config.infrastructure.optuna_n_jobs,
        help="Number of parallel jobs for Optuna optimization",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=experiment_config.infrastructure.optuna_timeout,
        help="Timeout in seconds for the optimization",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=experiment_config.infrastructure.optuna_storage,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)",
    )

    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
