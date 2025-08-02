"""Base experiment runner and utility functions for training and evaluation."""

from datetime import datetime
import os
from typing import Callable
import logging

import mlflow
import mlflow.pytorch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import Logger, CSVLogger, TensorBoardLogger
import pandas as pd  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
import torch
from torch.utils.data import DataLoader

from src.config.experiment_config import ExperimentConfig
from src.datasets.swine import SwineWaveformDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def initialize_dirs(experiment_config: ExperimentConfig, verbose: bool = False) -> None:
    """Initialize directories for the experiment.
    Args:
        experiment_config: Configuration for the experiment
        verbose: Whether to print verbose output
    """
    # Create model directory if not exists
    if verbose:
        logger.info("Creating directories for the experiment")
    if not os.path.exists(experiment_config.training.models_directory):
        os.makedirs(experiment_config.training.models_directory)
    if verbose:
        logger.info(
            "Created model directory: %s", experiment_config.training.models_directory
        )
    # Create log directory if not exists
    if not os.path.exists(experiment_config.training.log_directory):
        os.makedirs(experiment_config.training.log_directory)
    if verbose:
        logger.info(
            "Created log directory: %s", experiment_config.training.log_directory
        )


def load_swine_datasets(
    experiment_config: ExperimentConfig,
    prune_invalid: bool = True,
    target_transform: Callable | None = lambda x: x.float(),
    verbose: bool = False,
) -> tuple[SwineWaveformDataset, SwineWaveformDataset]:
    """Load the swine datasets and split them into train and test sets.
    Args:
        experiment_config: Configuration for the experiment
        prune_invalid: Whether to prune invalid samples
        target_transform: Transform to apply to the target
        verbose: Whether to print verbose output
    Returns:
        tuple: Tuple containing the training and test datasets
    """
    # Load annotation file
    if verbose:
        logger.info("Loading swine datasets")
    annotation: pd.DataFrame = pd.read_csv(experiment_config.data.annotation_file)
    if verbose:
        logger.info(
            "Loaded annotation file: %s", experiment_config.data.annotation_file
        )
    # Split data into train and test
    train_annotation, test_annotation = train_test_split(
        annotation, test_size=0.2, random_state=experiment_config.training.random_seed
    )
    if verbose:
        logger.info("Split data into train and test sets")
    # Sort by Timestamp
    train_annotation = train_annotation.sort_values(by=["Timestamp"])
    test_annotation = test_annotation.sort_values(by=["Timestamp"])
    # Save to csv
    train_annotation.to_csv(experiment_config.data.train_annotation_file, index=False)
    test_annotation.to_csv(experiment_config.data.test_annotation_file, index=False)
    if verbose:
        logger.info("Saved train and test annotation files")
        logger.info(
            "Train annotation file: %s", experiment_config.data.train_annotation_file
        )
        logger.info(
            "Test annotation file: %s", experiment_config.data.test_annotation_file
        )
    # Load train and test dataset
    dataset_train_wave: SwineWaveformDataset = SwineWaveformDataset(
        annotations_file=experiment_config.data.train_annotation_file,
        data_dir=experiment_config.data.data_directory,
        num_classes=experiment_config.annotation.num_classes,
        sample_seconds=experiment_config.annotation.sample_seconds,
        sample_rate=experiment_config.data.sample_rate,
        prune_invalid=prune_invalid,
        target_transform=target_transform,
    )
    dataset_test_wave: SwineWaveformDataset = SwineWaveformDataset(
        annotations_file=experiment_config.data.test_annotation_file,
        data_dir=experiment_config.data.data_directory,
        num_classes=experiment_config.annotation.num_classes,
        sample_seconds=experiment_config.annotation.sample_seconds,
        sample_rate=experiment_config.data.sample_rate,
        prune_invalid=prune_invalid,
        target_transform=target_transform,
    )
    if verbose:
        logger.info("Loaded swine datasets")
    # Calculate mean and std for normalization
    train_mean_wave: torch.Tensor = torch.zeros(1)
    train_std_wave: torch.Tensor = torch.zeros(1)
    if (
        experiment_config.data.train_mean is None
        or experiment_config.data.train_std is None
    ):
        if verbose:
            logger.info("Calculating mean and std for normalization")
        # Calculate mean and std for normalization
        for waveform, _ in dataset_train_wave:  # type: ignore
            train_mean_wave += waveform.mean()
            train_std_wave += waveform.std()
        train_mean_wave /= len(dataset_train_wave)
        train_std_wave /= len(dataset_train_wave)
    else:
        if verbose:
            logger.info("Using provided mean and std for normalization")
        train_mean_wave = torch.tensor(experiment_config.data.train_mean).float()
        train_std_wave = torch.tensor(experiment_config.data.train_std).float()
    if verbose:
        logger.info("Train mean (wave): %s", train_mean_wave)
        logger.info("Train std (wave): %s", train_std_wave)
    # Normalize datasets (update transform)
    dataset_train_wave.transform = lambda x: ((x - train_mean_wave) / train_std_wave)
    dataset_test_wave.transform = lambda x: ((x - train_mean_wave) / train_std_wave)
    return dataset_train_wave, dataset_test_wave


def create_data_loaders(
    experiment_config: ExperimentConfig,
    dataset_train_wave: SwineWaveformDataset,
    dataset_test_wave: SwineWaveformDataset,
    verbose: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Create data loaders for the training and test datasets.
    Args:
        experiment_config: Configuration for the experiment
        dataset_train_wave: Training dataset
        dataset_test_wave: Test dataset
        verbose: Whether to print verbose output
    Returns:
        tuple: Tuple containing the training and test data loaders
    """
    if verbose:
        logger.info("Creating data loaders")
    train_dataloader_wave = DataLoader(
        dataset_train_wave,
        batch_size=experiment_config.training.batch_size,
        shuffle=True,
        num_workers=experiment_config.training.num_workers,
        pin_memory=experiment_config.training.pin_memory,
        prefetch_factor=experiment_config.training.prefetch_factor,
    )
    test_dataloader_wave = DataLoader(
        dataset_test_wave,
        batch_size=experiment_config.training.batch_size,
        shuffle=False,
        num_workers=experiment_config.training.num_workers,
        pin_memory=experiment_config.training.pin_memory,
        prefetch_factor=experiment_config.training.prefetch_factor,
    )
    if verbose:
        logger.info("Created data loaders")
    return train_dataloader_wave, test_dataloader_wave


class ExperimentRunner(object):
    """General experiment runner for PyTorch Lightning models."""

    def __init__(
        self,
        experiment_config: ExperimentConfig,
        experiment_prefix: str,
        verbose: bool = False,
    ) -> None:
        """Initialize experiment runner.

        Args:
            experiment_config: Configuration for the experiment
            experiment_prefix: Prefix for the experiment name
            verbose: Whether to print verbose output
        """
        self.experiment_config = experiment_config
        self.experiment_prefix = experiment_prefix
        self.verbose = verbose

        # Initialize directories
        initialize_dirs(self.experiment_config, verbose=self.verbose)

        # Set up MLflow if enabled
        if self.experiment_config.infrastructure.use_mlflow:
            mlflow.set_tracking_uri(
                uri=self.experiment_config.infrastructure.mlflow_tracking_uri
            )
            mlflow.set_experiment(self.experiment_prefix)

    def _create_model(
        self, model_class: type[pl.LightningModule]
    ) -> pl.LightningModule:
        """Create a model of the specified class from the configuration.

        Args:
            model_class: The PyTorch Lightning model class to instantiate

        Returns:
            Instantiated model
        """
        return model_class(self.experiment_config.model)

    def _create_denoiser(
        self, denoiser_class: type[pl.LightningModule]
    ) -> pl.LightningModule:
        """Create a model of the specified class from the configuration.

        Args:
            model_class: The PyTorch Lightning model class to instantiate

        Returns:
            Instantiated model
        """
        return denoiser_class(self.experiment_config.denoiser)

    def _load_checkpoint(
        self, model_class: type[pl.LightningModule], checkpoint_file: str
    ) -> pl.LightningModule:
        """Load a model from a checkpoint file.

        Args:
            model_class: The PyTorch Lightning model class to instantiate
            checkpoint_file: Path to checkpoint file

        Returns:
            Model loaded from checkpoint
        """
        self.experiment_config.model.checkpoint_path = checkpoint_file
        return model_class.load_from_checkpoint(
            checkpoint_path=checkpoint_file,
            config=self.experiment_config.model,
        )

    def _find_checkpoint(self, run_name: str) -> str | None:
        """Find the latest checkpoint file for the given experiment name.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Path to checkpoint file or None if not found
        """
        checkpoint_file: str | None = None
        for file in os.listdir(self.experiment_config.training.models_directory):
            if file.startswith(run_name) and file.endswith(".ckpt"):
                if checkpoint_file is None or file > checkpoint_file:
                    checkpoint_file = file

        if checkpoint_file is not None:
            return os.path.join(
                self.experiment_config.training.models_directory, checkpoint_file
            )
        return None

    def _setup_loggers(self, experiment_name: str) -> list[Logger]:
        """Set up loggers for the experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            List of loggers
        """
        loggers_list: list[Logger] = [
            CSVLogger(
                save_dir=self.experiment_config.training.log_directory,
                name=experiment_name,
            ),
            TensorBoardLogger(
                save_dir=self.experiment_config.training.log_directory,
                name=experiment_name,
            ),
        ]

        return loggers_list

    def _setup_callbacks(self, experiment_name: str) -> list[pl.Callback]:
        """Set up callbacks for the experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            List of callbacks
        """
        checkpoint_callback: ModelCheckpoint = ModelCheckpoint(
            dirpath=self.experiment_config.training.models_directory,
            filename=experiment_name + "-{val_auroc:.2f}-{val_loss:.2f}-{epoch:02d}",
            monitor="val_auroc",
            verbose=True,
            save_top_k=1,
            save_weights_only=False,
            mode="max",
            auto_insert_metric_name=True,
            every_n_epochs=1,
            save_on_train_epoch_end=False,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_auroc",
            mode="max",
            patience=self.experiment_config.training.early_stopping_patience,
            verbose=True,
        )

        return [checkpoint_callback, early_stopping_callback]

    def _setup_mlflow_tracking(self) -> None:
        """Set up MLflow tracking."""
        if not self.experiment_config.infrastructure.use_mlflow:
            return

        mlflow.pytorch.autolog(
            log_models=self.experiment_config.infrastructure.mlflow_log_models,
            log_datasets=self.experiment_config.infrastructure.mlflow_log_datasets,
            checkpoint=self.experiment_config.infrastructure.mlflow_checkpoint,
            checkpoint_monitor=self.experiment_config.infrastructure.mlflow_checkpoint_monitor,
            checkpoint_mode=self.experiment_config.infrastructure.mlflow_checkpoint_mode,
            checkpoint_save_best_only=(
                self.experiment_config.infrastructure.mlflow_checkpoint_save_best_only
            ),
            checkpoint_save_weights_only=(
                self.experiment_config.infrastructure.mlflow_checkpoint_save_weights_only
            ),
            checkpoint_save_freq=(
                self.experiment_config.infrastructure.mlflow_checkpoint_save_freq
            ),
        )

    @staticmethod
    def _get_dataset_type(dataloader: DataLoader) -> str:
        """Get the dataset type from the dataloader.

        Args:
            dataloader: DataLoader to get dataset type from

        Returns:
            Dataset type as string ('wave', 'spec', or 'mfcc')
        """
        dataset_type: str = dataloader.dataset.__class__.__name__
        dataset_type_map = {
            "SwineWaveformDataset": "wave",
            "SwineSpectrogramDataset": "spec",
            "SwineMFCCDataset": "mfcc",
        }
        return dataset_type_map.get(dataset_type, "unknown")

    def _create_run_name_name(
        self, model_class: str, denoiser_class: str, dataset_type: str
    ) -> str:
        """Create an experiment name based on configuration and dataset type.

        Args:
            dataset_type: type of dataset ('wave', 'spec', or 'mfcc')

        Returns:
            Experiment name
        """
        return (
            f"{self.experiment_prefix}_"
            f"{model_class}_"
            f"{denoiser_class}_"
            f"{dataset_type}_{self.experiment_config.annotation.num_classes}-classes_"
            f"{self.experiment_config.data.num_bands}-bands"
        )

    def _prepare_model(
        self, model_class: type[pl.LightningModule], run_name: str
    ) -> pl.LightningModule | None:
        """Prepare model, either by creating a new one or loading from checkpoint.

        Args:
            model_class: The PyTorch Lightning model class to use.
            run_name: Name of the current run.

        Returns:
            The prepared model, or None if training should be skipped.
        """
        model = self._create_model(model_class)

        if (
            self.experiment_config.training.use_pretrained
            or self.experiment_config.evaluation.skip_trained_models
        ):
            checkpoint_file = self._find_checkpoint(run_name)
            if checkpoint_file is not None:
                if self.verbose:
                    logger.info("Found checkpoint %s", checkpoint_file)

                if self.experiment_config.evaluation.skip_trained_models:
                    if self.verbose:
                        logger.info("Skipping %s", run_name)
                    # Indicate skipping
                    return None

                model = self._load_checkpoint(model_class, checkpoint_file)
        return model

    def _train_model(
        self,
        model: pl.LightningModule,
        run_name: str,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
    ) -> None:
        """Train the model with the given data.

        Args:
            model: The model to train.
            run_name: Name of the current run.
            train_dataloader: DataLoader for training data.
            test_dataloader: DataLoader for test/validation data.
        """
        trainer_loggers = self._setup_loggers(run_name)
        callbacks = self._setup_callbacks(run_name)

        trainer: pl.Trainer = pl.Trainer(
            callbacks=callbacks,
            max_epochs=self.experiment_config.training.max_epochs,
            logger=trainer_loggers,
            log_every_n_steps=min(50, len(train_dataloader)),
        )

        self._setup_mlflow_tracking()

        if self.experiment_config.infrastructure.use_mlflow:
            with mlflow.start_run(
                run_name=f"{run_name}_{datetime.now()}",
                tags={"owner": "André Moreira Souza"},
                log_system_metrics=(
                    self.experiment_config.infrastructure.mlflow_log_system_metrics
                ),
            ):
                mlflow.log_param("experiment_name", run_name)
                mlflow.log_params(self.experiment_config.to_dict(flat=True))
                trainer.fit(model, train_dataloader, test_dataloader)
        else:
            trainer.fit(model, train_dataloader, test_dataloader)

    def run_experiment(
        self,
        model_class: type[pl.LightningModule],
        denoiser_class: type[pl.LightningModule],
    ) -> None:
        """Run the experiment with the given model class.

        Args:
            model_class: The PyTorch Lightning model class to use
        """
        # Load datasets
        dataset_train_wave, dataset_test_wave = load_swine_datasets(
            self.experiment_config, verbose=self.verbose
        )

        # Create data loaders
        train_dataloader, test_dataloader = create_data_loaders(
            self.experiment_config,
            dataset_train_wave,
            dataset_test_wave,
            verbose=self.verbose,
        )

        denoiser = self._create_denoiser(denoiser_class)
        denoiser.eval()
        if self.verbose:
            logger.info("Using denoiser: %s", denoiser_class.__name__)
        train_dataloader.dataset.transform = denoiser
        test_dataloader.dataset.transform = denoiser

        # Get input shape
        input_shape = next(iter(train_dataloader))[0].shape
        if self.verbose:
            logger.info("Input shape: %s", input_shape)

        # Get dataset type and create experiment name
        dataset_type = self._get_dataset_type(train_dataloader)
        run_name = self._create_run_name_name(
            model_class=model_class.__name__,
            denoiser_class=denoiser_class.__name__,
            dataset_type=dataset_type,
        )

        if self.verbose:
            logger.info("Training %s", run_name)

        # Prepare model (create or load)
        model = self._prepare_model(model_class, run_name)

        if model is None:
            # Model preparation indicated skipping
            return

        # Train the model
        self._train_model(model, run_name, train_dataloader, test_dataloader)

        if self.verbose:
            logger.info("Finished training %s", run_name)
