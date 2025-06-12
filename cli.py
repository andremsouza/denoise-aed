#!/usr/bin/env python
"""
Command-line interface for running intrinsic dimension experiments.
"""

import argparse
import logging

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

from src.architectures.denoise import kalman, sdrom, spectral_subtraction, identity

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_REGISTRY = {
    "ast": PLAST,
    "cnn14": PLCnn14,
    "dainet19": PLDaiNet19,
    "leenet24": PLLeeNet24,
    "mobilenetv1": PLMobileNetV1,
    "mobilenetv2": PLMobileNetV2,
    "resnet38": PLResNet38,
}

DENOISE_REGISTRY: dict[str, type] = {
    "kalman": kalman.AdaptiveKalman,
    "sdrom": sdrom.SDROM,
    "spectral_substraction": spectral_subtraction.SpectralSubtraction,
    "identity": identity.IdentityDenoiser,
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

    # Update denoiser config
    experiment_config.denoiser.process_variance = args.process_variance
    experiment_config.denoiser.initial_measurement_noise = (
        args.initial_measurement_noise
    )
    experiment_config.denoiser.adaptation_interval = args.adaptation_interval
    experiment_config.denoiser.window_size = args.denoiser_window_size
    experiment_config.denoiser.hop_size = args.denoiser_hop_size
    experiment_config.denoiser.noise_reduction_factor = args.noise_reduction_factor
    experiment_config.denoiser.noise_window_duration = args.noise_window_duration
    experiment_config.denoiser.alpha = args.alpha
    experiment_config.denoiser.beta = args.beta
    experiment_config.denoiser.adaptive = args.adaptive
    experiment_config.denoiser.sample_rate = args.sample_rate

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
        # Use setattr to safely set attributes or update them if they exist
        setattr(model_config, "imagenet_pretrain", args.ast_imagenet_pretrain)
        setattr(model_config, "audioset_pretrain", args.ast_audioset_pretrain)
        setattr(model_config, "model_size", args.ast_model_size)

    # Assign model config to experiment config
    experiment_config.model = model_config

    # Set high precision for float32 matrix multiplication
    torch.set_float32_matmul_precision("high")

    # Run the experiment
    experiment_prefix: str = "denoising_experiments"
    logger.info("Starting experiment with prefix: %s", experiment_prefix)
    logger.info("Using model type: %s", args.model_type)
    logger.info("Arguments: %s", args)

    try:
        # Check if optimization is enabled
        if args.optimize or args.optimize_only:
            # Create OptunaExperimentRunner
            optuna_experiment_runner = OptunaExperimentRunner(
                experiment_config=experiment_config,
                experiment_prefix=experiment_prefix,
                optimization_metric=args.optimize_metric,
                direction=args.optimize_direction,
                n_trials=args.n_trials,
                n_jobs=args.n_jobs,
                timeout=args.timeout,
                study_name=args.study_name if args.study_name else experiment_prefix,
                storage=args.storage,
                verbose=True,
            )

            # Run optimization and then final experiment with best parameters
            if args.optimize_only:
                # Just run optimization
                optuna_experiment_runner.run_optimization(
                    model_class=MODELS_REGISTRY[args.model_type],
                    denoiser_class=DENOISE_REGISTRY[args.denoiser],
                )
            else:
                # Run optimization and then final experiment
                optuna_experiment_runner.run_with_best_params(
                    model_class=MODELS_REGISTRY[args.model_type],
                    denoiser_class=DENOISE_REGISTRY[args.denoiser],
                )
        else:
            # Create regular ExperimentRunner
            regular_experiment_runner = ExperimentRunner(
                experiment_config=experiment_config,
                experiment_prefix=experiment_prefix,
                verbose=True,
            )
            regular_experiment_runner.run_experiment(
                model_class=MODELS_REGISTRY[args.model_type],
                denoiser_class=DENOISE_REGISTRY[args.denoiser],
            )
    except KeyError:
        logger.error("Model type %s not supported yet.", args.model_type)
        return

    logger.info("Experiment complete")


def _add_infrastructure_args(
    parser: argparse.ArgumentParser, experiment_config: ExperimentConfig
) -> None:
    """Add infrastructure-related arguments to the parser.

    Args:
        parser: The argument parser
        experiment_config: The experiment configuration
    """
    infra_args = parser.add_argument_group("Infrastructure Arguments")
    infra_args.add_argument(
        "--use-mlflow",
        action="store_true",
        default=experiment_config.infrastructure.use_mlflow,
        help="Use MLflow for experiment tracking",
    )


def _add_data_args(
    parser: argparse.ArgumentParser, experiment_config: ExperimentConfig
) -> None:
    """Add data-related arguments to the parser.

    Args:
        parser: The argument parser
        experiment_config: The experiment configuration
    """
    data_args = parser.add_argument_group("Data Arguments")
    data_args.add_argument(
        "--data-dir",
        type=str,
        default=experiment_config.data.data_directory,
        help="Directory containing audio data",
    )
    data_args.add_argument(
        "--annotation-file",
        type=str,
        default=experiment_config.data.annotation_file,
        help="Path to annotation file",
    )
    data_args.add_argument(
        "--sample-rate",
        type=int,
        default=experiment_config.data.sample_rate,
        help="Audio sample rate",
    )
    data_args.add_argument(
        "--num-bands",
        type=int,
        default=experiment_config.data.num_bands,
        help="Number of frequency bands",
    )


def _add_denoising_args(
    parser: argparse.ArgumentParser, experiment_config: ExperimentConfig
) -> None:
    denoising_args = parser.add_argument_group("Denoiser Arguments")
    denoising_args.add_argument(
        "--denoiser",
        "-d",
        type=str,
        choices=list(DENOISE_REGISTRY.keys()),
        required=True,
        help="Type of denoiser to use",
    )
    denoising_args.add_argument(
        "--process-variance",
        type=float,
        default=experiment_config.denoiser.process_variance,
        help="Process variance for the denoiser (default: 1e-5)",
    )
    denoising_args.add_argument(
        "--initial-measurement-noise",
        type=float,
        default=experiment_config.denoiser.initial_measurement_noise,
        help="Initial measurement noise for the denoiser (default: 1e-4)",
    )
    denoising_args.add_argument(
        "--adaptation-interval",
        type=int,
        default=experiment_config.denoiser.adaptation_interval,
        help="Adaptation interval for the denoiser (default: 100)",
    )
    denoising_args.add_argument(
        "--denoiser-window-size",
        type=int,
        default=experiment_config.denoiser.window_size,
        help="Window size for the denoiser (default: 1024)",
    )
    denoising_args.add_argument(
        "--denoiser-hop-size",
        type=int,
        default=experiment_config.denoiser.hop_size,
        help="Hop size for the denoiser (default: 512)",
    )
    denoising_args.add_argument(
        "--noise-reduction-factor",
        type=float,
        default=experiment_config.denoiser.noise_reduction_factor,
        help="Noise reduction factor for the denoiser (default: 1.0)",
    )
    denoising_args.add_argument(
        "--noise-window-duration",
        type=float,
        default=experiment_config.denoiser.noise_window_duration,
        help="Noise window duration for the denoiser in seconds (default: 0.1)",
    )
    denoising_args.add_argument(
        "--alpha",
        type=float,
        default=experiment_config.denoiser.alpha,
        help="Alpha parameter for the denoiser (default: 0.95)",
    )
    denoising_args.add_argument(
        "--beta",
        type=float,
        default=experiment_config.denoiser.beta,
        help="Beta parameter for the denoiser (default: 0.98)",
    )
    denoising_args.add_argument(
        "--adaptive",
        action=argparse.BooleanOptionalAction,
        default=experiment_config.denoiser.adaptive,
        help="Enable adaptive mode for the denoiser (default: True)",
    )


def _add_training_args(
    parser: argparse.ArgumentParser, experiment_config: ExperimentConfig
) -> None:
    """Add training-related arguments to the parser.

    Args:
        parser: The argument parser
        experiment_config: The experiment configuration
    """
    training_args = parser.add_argument_group("Training Arguments")
    training_args.add_argument(
        "--batch-size",
        type=int,
        default=experiment_config.training.batch_size,
        help="Training batch size",
    )
    training_args.add_argument(
        "--models-dir",
        type=str,
        default=experiment_config.training.models_directory,
        help="Directory to save model checkpoints",
    )
    training_args.add_argument(
        "--logs-dir",
        type=str,
        default=experiment_config.training.log_directory,
        help="Directory to save training logs",
    )
    training_args.add_argument(
        "--max-epochs",
        type=int,
        default=experiment_config.training.max_epochs,
        help="Maximum number of training epochs",
    )
    training_args.add_argument(
        "--patience",
        type=int,
        default=experiment_config.training.early_stopping_patience,
        help="Patience for early stopping",
    )
    training_args.add_argument(
        "--num-workers",
        type=int,
        default=experiment_config.training.num_workers,
        help="Number of workers for data loading",
    )
    training_args.add_argument(
        "--device",
        type=str,
        default=experiment_config.training.device,
        help="Device to use for training (cuda or cpu)",
    )
    training_args.add_argument(
        "--seed",
        type=int,
        default=experiment_config.training.random_seed,
        help="Random seed for reproducibility",
    )
    training_args.add_argument(
        "--use-pretrained",
        "-p",
        action="store_true",
        default=experiment_config.training.use_pretrained,
        help="Use pretrained model checkpoints if available",
    )


def _add_annotation_args(
    parser: argparse.ArgumentParser, experiment_config: ExperimentConfig
) -> None:
    """Add annotation-related arguments to the parser.

    Args:
        parser: The argument parser
        experiment_config: The experiment configuration
    """
    annotation_args = parser.add_argument_group("Annotation Arguments")
    annotation_args.add_argument(
        "--num-classes",
        type=int,
        default=experiment_config.annotation.num_classes,
        help="Number of classes for classification",
    )
    annotation_args.add_argument(
        "--sample-seconds",
        type=float,
        default=experiment_config.annotation.sample_seconds,
        help="Length of audio samples in seconds",
    )


def _add_evaluation_args(
    parser: argparse.ArgumentParser, experiment_config: ExperimentConfig
) -> None:
    """Add evaluation-related arguments to the parser.

    Args:
        parser: The argument parser
        experiment_config: The experiment configuration
    """
    evaluation_args = parser.add_argument_group("Evaluation Arguments")
    evaluation_args.add_argument(
        "--skip-trained",
        action="store_true",
        default=experiment_config.evaluation.skip_trained_models,
        help="Skip already trained models",
    )
    evaluation_args.add_argument(
        "--pred-threshold",
        type=float,
        default=experiment_config.evaluation.pred_threshold,
        help="Threshold for making predictions",
    )


def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model selection arguments to the parser.

    Args:
        parser: The argument parser
    """
    model_args = parser.add_argument_group("Model Arguments")
    model_args.add_argument(
        "--model-type",
        "-m",
        type=str,
        required=True,
        choices=list(MODELS_REGISTRY.keys()),
        help="Type of model architecture to use",
    )


def _add_model_specific_args(
    parser: argparse.ArgumentParser, experiment_config: ExperimentConfig
) -> None:
    """Add model-specific arguments to the parser.

    Args:
        parser: The argument parser
        experiment_config: The experiment configuration
    """
    model_specific_args = parser.add_argument_group("Model-Specific Arguments")
    model_specific_args.add_argument(
        "--lr",
        type=float,
        default=(
            experiment_config.model.learning_rate
            if hasattr(experiment_config.model, "learning_rate")
            else 1e-4
        ),
        help="Learning rate for optimizer",
    )
    model_specific_args.add_argument(
        "--weight-decay",
        type=float,
        default=(
            experiment_config.model.weight_decay
            if hasattr(experiment_config.model, "weight_decay")
            else 1e-2
        ),
        help="Weight decay for optimizer",
    )

    # AST-specific arguments
    model_specific_args.add_argument(
        "--ast-imagenet-pretrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ImageNet pretrained weights for AST (default: True)",
    )
    model_specific_args.add_argument(
        "--ast-audioset-pretrain",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use AudioSet pretrained weights for AST (default: True)",
    )
    model_specific_args.add_argument(
        "--ast-model-size",
        type=str,
        default="base384",
        choices=["base384", "small384"],
        help="Model size for AST",
    )


def _add_optuna_args(
    parser: argparse.ArgumentParser, experiment_config: ExperimentConfig
) -> None:
    """Add Optuna-related arguments to the parser.

    Args:
        parser: The argument parser
        experiment_config: The experiment configuration
    """
    optuna_args = parser.add_argument_group("Optuna Arguments")
    optuna_args.add_argument(
        "--optimize",
        action="store_true",
        help="Enable hyperparameter optimization with Optuna",
    )
    optuna_args.add_argument(
        "--optimize-only",
        action="store_true",
        help="Run only optimization without final experiment",
    )
    optuna_args.add_argument(
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
    optuna_args.add_argument(
        "--optimize-direction",
        type=str,
        default=experiment_config.infrastructure.optuna_direction,
        choices=["maximize", "minimize"],
        help="Direction of optimization",
    )
    optuna_args.add_argument(
        "--n-trials",
        type=int,
        default=experiment_config.infrastructure.optuna_n_trials,
        help="Number of optimization trials to run",
    )
    optuna_args.add_argument(
        "--n-jobs",
        type=int,
        default=experiment_config.infrastructure.optuna_n_jobs,
        help="Number of parallel jobs for Optuna optimization",
    )
    optuna_args.add_argument(
        "--timeout",
        type=int,
        default=experiment_config.infrastructure.optuna_timeout,
        help="Timeout in seconds for the optimization",
    )
    optuna_args.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name of the Optuna study",
    )
    optuna_args.add_argument(
        "--storage",
        type=str,
        default=experiment_config.infrastructure.optuna_storage,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)",
    )


def parse_args(experiment_config: ExperimentConfig) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        experiment_config: Default configuration to use for argument defaults

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Run intrinsic dimension experiments")

    # Add arguments by category
    _add_infrastructure_args(parser, experiment_config)
    _add_data_args(parser, experiment_config)
    _add_training_args(parser, experiment_config)
    _add_denoising_args(parser, experiment_config)
    _add_annotation_args(parser, experiment_config)
    _add_evaluation_args(parser, experiment_config)
    _add_model_args(parser)
    _add_model_specific_args(parser, experiment_config)
    _add_optuna_args(parser, experiment_config)

    # Parse arguments
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
