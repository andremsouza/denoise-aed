"""Audio DataModule for experiments with BaseAudioLightningModule architectures."""

from typing import Callable, Literal

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from src.config.experiment_config import ExperimentConfig
from src.datasets.swine import SwineWaveformDataset, SwineSpectrogramDataset


class SwineAudioDataModule(pl.LightningDataModule):
    """General Audio DataModule for experiments with various audio architectures."""

    def __init__(
        self,
        config: ExperimentConfig,
        dataset_class: Literal["waveform", "spectrogram"] = "waveform",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        annotations_file: str | pd.DataFrame | None = None,
        data_dir: str | None = None,
    ) -> None:
        """Initialize AudioDataModule using configuration dataclasses.

        Args:
            config: ExperimentConfig instance containing all configuration parameters
            dataset_class: Type of dataset to use ('waveform' or 'spectrogram')
            transform: Transform to apply to samples
            target_transform: Transform to apply to targets
            annotations_file: Optional override for annotations file path (uses config if None)
            data_dir: Optional override for data directory path (uses config if None)
        """
        super().__init__()

        # Use parameters from config but allow for optional overrides
        self.annotations_file = annotations_file or config.data.annotation_file
        self.data_dir = data_dir or config.data.data_directory

        # Configuration parameters
        self.dataset_class = dataset_class
        self.num_classes = config.annotation.num_classes
        self.sample_seconds = config.annotation.sample_seconds
        self.sample_rate = config.data.sample_rate
        self.batch_size = config.training.batch_size

        # Default split ratios (70/15/15)
        self.train_ratio = config.data.train_ratio
        self.val_ratio = config.data.val_ratio
        self.test_ratio = config.data.test_ratio

        # Verify ratios sum to 1
        assert (
            abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6
        ), "Train, val, and test ratios must sum to 1"

        # DataLoader parameters
        self.num_workers = config.training.num_workers
        self.pin_memory = config.training.pin_memory
        self.persistent_workers = self.num_workers > 0
        self.prefetch_factor = config.training.prefetch_factor

        # Transforms and seed
        self.transform = transform
        self.target_transform = target_transform
        self.seed = config.training.random_seed

        # Save hyperparameters
        self.save_hyperparameters(ignore=["config", "transform", "target_transform"])

        # Initialized in setup()
        self.dims = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _create_dataset(self) -> Dataset:
        """Create the dataset instance based on configuration.

        Returns:
            Dataset: The created dataset
        """
        if self.dataset_class == "waveform":
            return SwineWaveformDataset(
                annotations_file=self.annotations_file,
                data_dir=self.data_dir,
                num_classes=self.num_classes,
                sample_seconds=self.sample_seconds,
                sample_rate=self.sample_rate,
                prune_invalid=True,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        if self.dataset_class == "spectrogram":
            return SwineSpectrogramDataset(
                annotations_file=self.annotations_file,
                data_dir=self.data_dir,
                num_classes=self.num_classes,
                sample_seconds=self.sample_seconds,
                sample_rate=self.sample_rate,
                prune_invalid=True,
                transform=self.transform,
                target_transform=self.target_transform,
            )
        raise ValueError(f"Unknown dataset class: {self.dataset_class}")

    def setup(self, stage: str | None = None) -> None:
        """Set up datasets for training, validation, and testing.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Create full dataset
        dataset = self._create_dataset()

        # Calculate split sizes
        dataset_size = len(dataset)
        train_size = int(dataset_size * self.train_ratio)
        val_size = int(dataset_size * self.val_ratio)
        test_size = dataset_size - train_size - val_size

        # Set random seed for reproducibility
        generator = torch.Generator().manual_seed(self.seed)

        # Split dataset
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

        self.dims = dataset[0][0].shape

    def _create_dataloader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        """Create a DataLoader for the given dataset.

        Args:
            dataset: Dataset to create DataLoader for
            shuffle: Whether to shuffle the data

        Returns:
            DataLoader: The created DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            worker_init_fn=self._worker_init_fn,
        )

    def _worker_init_fn(self, worker_id: int) -> None:
        """Initialize workers with different seeds.

        Args:
            worker_id: ID of the worker
        """
        worker_seed = self.seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def train_dataloader(self) -> DataLoader:
        """Create the training DataLoader.

        Returns:
            DataLoader: Training DataLoader
        """
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create the validation DataLoader.

        Returns:
            DataLoader: Validation DataLoader
        """
        return self._create_dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        """Create the test DataLoader.

        Returns:
            DataLoader: Test DataLoader
        """
        return self._create_dataloader(self.test_dataset)

    def predict_dataloader(self) -> DataLoader:
        """Create the prediction DataLoader.

        Returns:
            DataLoader: Prediction DataLoader
        """
        return self._create_dataloader(self.test_dataset)

    def teardown(self, stage: str | None = None) -> None:
        """Clean up after a stage is finished.

        Args:
            stage: Current stage ('fit', 'validate', 'test', or 'predict')
        """
        # Clean up any resources and memory
        torch.cuda.empty_cache()

        # Clear datasets to free memory
        if stage == "fit" or stage is None:
            self.train_dataset = None
            self.val_dataset = None
            self.test_dataset = None
