from datetime import datetime
import os
from typing import Callable

import mlflow
import mlflow.pytorch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import Logger, CSVLogger, TensorBoardLogger, MLFlowLogger
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from src.config.experiment_config import ExperimentConfig
from src.datasets.swine import SwineWaveformDataset


def initialize_dirs(experiment_config: ExperimentConfig, verbose: bool = False) -> None:
    """Initialize directories for the experiment.
    Args:
        experiment_config: Configuration for the experiment
        verbose: Whether to print verbose output
    """
    # Create model directory if not exists
    if verbose:
        print(f"[{datetime.now()}]: Creating directories for the experiment")
    if not os.path.exists(experiment_config.training.models_directory):
        os.makedirs(experiment_config.training.models_directory)
    if verbose:
        print(
            f"[{datetime.now()}]: Created model directory:"
            f"{experiment_config.training.models_directory}"
        )
    # Create log directory if not exists
    if not os.path.exists(experiment_config.training.log_directory):
        os.makedirs(experiment_config.training.log_directory)
    if verbose:
        print(
            f"[{datetime.now()}]: Created log directory: {experiment_config.training.log_directory}"
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
        print(f"[{datetime.now()}]: Loading swine datasets")
    annotation: pd.DataFrame = pd.read_csv(experiment_config.data.annotation_file)
    if verbose:
        print(
            f"[{datetime.now()}]: Loaded annotation file: {experiment_config.data.annotation_file}"
        )
    # Split data into train and test
    train_annotation, test_annotation = train_test_split(
        annotation, test_size=0.2, random_state=experiment_config.training.random_seed
    )
    if verbose:
        print(f"[{datetime.now()}]: Split data into train and test sets")
    # Sort by Timestamp
    train_annotation = train_annotation.sort_values(by=["Timestamp"])
    test_annotation = test_annotation.sort_values(by=["Timestamp"])
    # Save to csv
    train_annotation.to_csv(experiment_config.data.train_annotation_file, index=False)
    test_annotation.to_csv(experiment_config.data.test_annotation_file, index=False)
    if verbose:
        print(f"[{datetime.now()}]: Saved train and test annotation files")
        print(f"Train annotation file: {experiment_config.data.train_annotation_file}")
        print(f"Test annotation file: {experiment_config.data.test_annotation_file}")
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
        print(f"[{datetime.now()}]: Loaded swine datasets")
    # Calculate mean and std for normalization
    train_mean_wave: torch.Tensor = torch.zeros(1)
    train_std_wave: torch.Tensor = torch.zeros(1)
    if (
        experiment_config.data.train_mean is None
        or experiment_config.data.train_std is None
    ):
        if verbose:
            print(f"[{datetime.now()}]: Calculating mean and std for normalization")
        # Calculate mean and std for normalization
        for waveform, _ in dataset_train_wave:  # type: ignore
            train_mean_wave += waveform.mean()
            train_std_wave += waveform.std()
        train_mean_wave /= len(dataset_train_wave)
        train_std_wave /= len(dataset_train_wave)
    else:
        if verbose:
            print(f"[{datetime.now()}]: Using provided mean and std for normalization")
        train_mean_wave = torch.tensor(experiment_config.data.train_mean).float()
        train_std_wave = torch.tensor(experiment_config.data.train_std).float()
    if verbose:
        print(f"[{datetime.now()}]: Train mean (wave): {train_mean_wave}")
        print(f"[{datetime.now()}]: Train std (wave): {train_std_wave}")
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
        print(f"[{datetime.now()}]: Creating data loaders")
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
        print(f"[{datetime.now()}]: Created data loaders")
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
        loggers: list[Logger] = [
            CSVLogger(
                save_dir=self.experiment_config.training.log_directory,
                name=experiment_name,
            ),
            TensorBoardLogger(
                save_dir=self.experiment_config.training.log_directory,
                name=experiment_name,
            ),
        ]

        return loggers

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
            checkpoint_save_best_only=self.experiment_config.infrastructure.mlflow_checkpoint_save_best_only,
            checkpoint_save_weights_only=self.experiment_config.infrastructure.mlflow_checkpoint_save_weights_only,
            checkpoint_save_freq=self.experiment_config.infrastructure.mlflow_checkpoint_save_freq,
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
        # Transform type to string in ["wave", "spec", "mfcc"]
        dataset_type_map = {
            "SwineWaveformDataset": "wave",
            "SwineSpectrogramDataset": "spec",
            "SwineMFCCDataset": "mfcc",
        }
        return dataset_type_map.get(dataset_type, "unknown")

    def _create_run_name_name(self, model_class: str, dataset_type: str) -> str:
        """Create an experiment name based on configuration and dataset type.

        Args:
            dataset_type: type of dataset ('wave', 'spec', or 'mfcc')

        Returns:
            Experiment name
        """
        return (
            f"{self.experiment_prefix}_"
            f"{model_class}_"
            f"{dataset_type}_{self.experiment_config.annotation.num_classes}-classes_"
            f"{self.experiment_config.data.num_bands}-bands"
        )

    def _extract_embeddings(
        self,
        model: pl.LightningModule,
        dataloader: DataLoader,
        experiment_name: str,
        prefix: str,
        device: str | torch.device = "cpu",
    ) -> None:
        """Extract embeddings from a dataloader and save them to disk.

        Args:
            model: The PyTorch Lightning model with an extract_embeddings method.
            dataloader: DataLoader for the dataset (train/val/test).
            experiment_name: Name of the experiment.
            prefix: Prefix for the saved embeddings file.
            device: Device on which to perform computations.
        """
        device = torch.device(device)
        if self.verbose:
            print(
                f"[{datetime.now()}]: Extracting {prefix} embeddings for {experiment_name} on {device}"
            )
        model.to(device)
        model.eval()
        embeddings_list = []
        with torch.no_grad():
            for waveform, _ in dataloader:
                waveform = waveform.to(device)
                embeddings = model.extract_embeddings(waveform)
                embeddings_list.append(embeddings.cpu().numpy())
        embeddings_array = np.concatenate(embeddings_list, axis=0)
        save_path = os.path.join(
            self.experiment_config.training.models_directory,
            f"{experiment_name}_{prefix}_embeddings.npy",
        )
        np.save(save_path, embeddings_array)
        if self.verbose:
            print(f"[{datetime.now()}]: Saved {prefix} embeddings to {save_path}")

    def run_experiment(self, model_class: type[pl.LightningModule]) -> None:
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

        # Get input shape
        input_shape = next(iter(train_dataloader))[0].shape
        if self.verbose:
            print(f"Input shape: {input_shape}")

        # Get dataset type and create experiment name
        dataset_type = self._get_dataset_type(train_dataloader)
        run_name = self._create_run_name_name(
            model_class=model_class.__name__, dataset_type=dataset_type
        )

        if self.verbose:
            print(f"[{datetime.now()}]: Training {run_name}")

        # Create model or load from checkpoint
        model = self._create_model(model_class)

        # Check for existing checkpoint
        if (
            self.experiment_config.training.use_pretrained
            or self.experiment_config.evaluation.skip_trained_models
        ):
            checkpoint_file = self._find_checkpoint(run_name)
            if checkpoint_file is not None:
                if self.verbose:
                    print(f"[{datetime.now()}]: Found checkpoint {checkpoint_file}")

                if self.experiment_config.evaluation.skip_trained_models:
                    if self.verbose:
                        print(f"[{datetime.now()}]: Skipping {run_name}")
                    return

                # Load model from checkpoint
                model = self._load_checkpoint(model_class, checkpoint_file)

        # Set up loggers and callbacks
        loggers = self._setup_loggers(run_name)
        callbacks = self._setup_callbacks(run_name)

        # 1. Extract embeddings from pretrained model
        # Extract embeddings for training and testing datasets using the new function
        self._extract_embeddings(
            model,
            train_dataloader,
            run_name,
            "train_pretrained",
            device=self.experiment_config.training.device,
        )
        self._extract_embeddings(
            model,
            test_dataloader,
            run_name,
            "test_pretrained",
            device=self.experiment_config.training.device,
        )
        # 2. Train the model

        # Set up trainer
        trainer: pl.Trainer = pl.Trainer(
            callbacks=callbacks,
            max_epochs=self.experiment_config.training.max_epochs,
            logger=loggers,
            log_every_n_steps=min(50, len(train_dataloader)),
        )

        # Setup MLflow tracking
        self._setup_mlflow_tracking()

        # Run training
        if self.experiment_config.infrastructure.use_mlflow:
            with mlflow.start_run(
                run_name=f"{run_name}_{datetime.now()}",
                tags={"owner": "André Moreira Souza"},
                log_system_metrics=self.experiment_config.infrastructure.mlflow_log_system_metrics,
            ):
                mlflow.log_param("experiment_name", run_name)
                mlflow.log_params(self.experiment_config.to_dict(flat=True))
                trainer.fit(model, train_dataloader, test_dataloader)
        else:
            trainer.fit(model, train_dataloader, test_dataloader)

        # 3. Extract embeddings from trained model
        self._extract_embeddings(
            model,
            train_dataloader,
            run_name,
            "train_finetuned",
            device=self.experiment_config.training.device,
        )
        self._extract_embeddings(
            model,
            test_dataloader,
            run_name,
            "test_finetuned",
            device=self.experiment_config.training.device,
        )

        if self.verbose:
            print(f"[{datetime.now()}]: Finished training {run_name}")
