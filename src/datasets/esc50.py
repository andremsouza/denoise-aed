"""Dataset for audio files with annotations of the ESC-50 dataset."""

# %%
import argparse
from collections import OrderedDict
from datetime import datetime
import os
import shutil
from typing import Callable
import urllib.request
import warnings
import zipfile

import librosa as lr
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# %% [markdown]
# # Constants (only for testing)

# %%

# %%


class ESC50WaveformDataset(Dataset):
    """Dataset for ESC-50 dataset."""

    ARCHIVE_URL = "https://github.com/karoldvl/ESC-50/archive/master.zip"

    def __init__(
        self,
        annotations_file: str | pd.DataFrame,
        data_dir: str,
        sample_seconds: float = 1.0,
        sample_rate: int = 16000,
        filter_classes: str | int | list[str] | list[int] | None = None,
        filter_folds: int | list[int] | None = None,
        prune_invalid: bool = False,
        download: bool = False,
        verbose: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            annotations_file (str | pd.DataFrame): path to annotations file or dataframe
            data_dir (str): path to audio files
            sample_seconds (float, optional): duration of samples in seconds.
                Defaults to 1.0.
            sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
            filter_classes (str | int | list[str] | list[int] | None, optional):
                classes to filter. Defaults to None.
            filter_folds (int | list[int] | None, optional): folds to filter.
                Defaults to None.
            prune_invalid (bool, optional): whether to prune invalid samples.
                Defaults to False.
            download (bool, optional): whether to download dataset. Defaults to False.
            verbose (bool, optional): whether to print progress. Defaults to True.
            transform (Callable, optional): transform to apply to samples.
                Defaults to None.
            target_transform (Callable, optional): transform to apply to labels.
                Defaults to None.
        """
        super().__init__()
        # Set attributes
        self.annotations_file: str | pd.DataFrame = annotations_file
        self.data_dir: str = data_dir
        self.sample_seconds: float = sample_seconds
        self.sample_rate: int = sample_rate
        self.filter_classes: str | int | list[str] | list[int] | None = filter_classes
        self.filter_folds: int | list[int] | None = filter_folds
        self.prune_invalid: bool = prune_invalid
        self.download: bool = download
        self.verbose: bool = verbose
        self.transform: Callable | None = transform
        self.target_transform: Callable | None = target_transform
        # Download dataset
        if self.download:
            self._download()
        # Load annotations
        if isinstance(self.annotations_file, str):
            self.annotations: pd.DataFrame = pd.read_csv(
                self.annotations_file, header=0
            )
        else:
            self.annotations = self.annotations_file
        # Filter annotations
        if self.filter_classes is not None:
            self._filter_classes()
        if self.filter_folds is not None:
            self._filter_folds()
        self.data = self._annotations_to_samples()
        self.num_classes: int = len(self.data["target"].unique())
        if self.prune_invalid:
            self._prune_invalid_samples()

    def _download(self) -> None:
        """Download ESC-50 dataset.

        Zip structure:
        ESC-50-master/
            meta/
                esc50.csv
            audio/
                1-100032-A-0.wav
                ...
        """
        # Create data directory if it does not exist
        if not os.path.exists(self.data_dir):
            if self.verbose:
                print(f"Creating directory {self.data_dir}...")
            os.makedirs(self.data_dir)
        # Check if annotation file's directory exists
        if isinstance(self.annotations_file, str):
            annotations_dir: str = os.path.dirname(self.annotations_file)
            if not os.path.exists(annotations_dir):
                if self.verbose:
                    print(f"Creating directory {annotations_dir}...")
                os.makedirs(annotations_dir)
        # Download dataset
        archive_path = os.path.join(self.data_dir, "ESC-50-master.zip")
        if os.path.exists(archive_path):
            # Check if archive is a valid zip file
            if self.verbose:
                print(f"Found archive at {archive_path}. Checking if it is valid...")
            try:
                with zipfile.ZipFile(archive_path) as zip_ref:
                    zip_ref.testzip()
                if self.verbose:
                    print(f"Archive at {archive_path} is valid.")
            except zipfile.BadZipFile:
                # Print warning
                if self.verbose:
                    print(f"Invalid archive at {archive_path}. Downloading again...")
                # Remove invalid archive
                os.remove(archive_path)
        if not os.path.exists(archive_path):
            if self.verbose:
                print("Downloading ESC-50 dataset...")
            urllib.request.urlretrieve(self.ARCHIVE_URL, archive_path)  # nosec  # noqa
            if self.verbose:
                print("Done downloading ESC-50 dataset.")
        # Extract dataset
        if self.verbose:
            print("Extracting ESC-50 dataset...")
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)
        if self.verbose:
            print(f"Done extracting ESC-50 dataset to {self.data_dir}.")
        # Move audio files to data_dir
        if self.verbose:
            print(f"Moving audio files to {self.data_dir}...")
        audio_dir = os.path.join(self.data_dir, "ESC-50-master", "audio")
        for file in os.listdir(audio_dir):
            shutil.copy(os.path.join(audio_dir, file), self.data_dir)
        # If annotations file is str, move it to data_dir
        if isinstance(self.annotations_file, str):
            if self.verbose:
                print(f"Moving annotations file to {self.annotations_file}...")
            shutil.copy(
                os.path.join(self.data_dir, "ESC-50-master", "meta", "esc50.csv"),
                self.annotations_file,
            )
        # Remove empty directory
        if self.verbose:
            print("Removing unnecessary files...")
        shutil.rmtree(os.path.join(self.data_dir, "ESC-50-master"))
        # Remove archive
        if self.verbose:
            print("Removing archive...")
        os.remove(archive_path)
        if self.verbose:
            print("Done downloading and extracting ESC-50 dataset.")

    def _filter_classes(self) -> None:
        if isinstance(self.filter_classes, str):
            self.filter_classes = [self.filter_classes]
        elif isinstance(self.filter_classes, int):
            self.filter_classes = [self.filter_classes]
        # Check if list of ints or strs
        if all(isinstance(item, int) for item in self.filter_classes):  # type: ignore
            # Transform to set
            self.filter_classes = set(self.filter_classes)  # type: ignore
            # Filter using "target"
            self.annotations = self.annotations[
                self.annotations["target"].isin(self.filter_classes)
            ]
        elif all(isinstance(item, str) for item in self.filter_classes):  # type: ignore
            # Transform to set
            self.filter_classes = set(self.filter_classes)  # type: ignore
            # Filter using "category"
            self.annotations = self.annotations[
                self.annotations["category"].isin(self.filter_classes)
            ]

    def _filter_folds(self) -> None:
        if isinstance(self.filter_folds, int):
            self.filter_folds = [self.filter_folds]
        # Convert to set
        self.filter_folds = set(self.filter_folds)  # type: ignore
        # Filter using "fold"
        self.annotations = self.annotations[
            self.annotations["fold"].isin(self.filter_folds)
        ]

    def _annotations_to_samples(self) -> pd.DataFrame:
        samples: list[OrderedDict[str, object]] | pd.DataFrame = []
        for _, row in self.annotations.iterrows():
            # Get audio file path
            audio_file_path: str = os.path.join(self.data_dir, row["filename"])
            # Calculate number of new samples
            duration: float = lr.get_duration(path=audio_file_path)
            num_samples: int = int(duration // self.sample_seconds)
            # Generate and append samples
            for i in range(num_samples):
                # ordered dict to preserve order
                sample = OrderedDict(
                    [
                        ("audio_file_path", audio_file_path),
                        ("offset", i * self.sample_seconds),
                        ("duration", self.sample_seconds),
                        ("fold", row["fold"]),
                        ("target", row["target"]),
                        ("category", row["category"]),
                    ]
                )
                samples.append(sample)  # type: ignore
        samples = pd.DataFrame(samples)
        return samples

    def _prune_invalid_samples(self) -> None:
        """Prune invalid samples, e.g., with invalid/insufficient size."""
        # Iterate over samples
        # If a sample has invalid size, remove it
        drop_indices: list = []
        # Try to get each sample
        for idx in range(len(self)):  # pylint: disable=consider-using-enumerate
            try:
                # Load audio
                self._load_audio(idx)
            except ValueError:
                # Add idx to list of indices to drop
                drop_indices.append(idx)
        # Drop invalid samples
        self.data.drop(drop_indices, inplace=True)
        # Reset index
        self.data.reset_index(drop=True, inplace=True)
        warnings.warn(f"Pruned {len(drop_indices)} invalid samples.")

    def _load_audio(self, idx: int, normalize: bool = True) -> tuple[np.ndarray, float]:
        """Load audio file at index idx.

        Args:
            idx (int): index of sample
            normalize (bool, optional): whether to normalize audio. Defaults to True.

        Returns:
            tuple[np.ndarray, int]: waveform and sample rate
        """
        # Get sample
        sample_data: pd.Series = self.data.iloc[idx]
        # Get audio file path
        audio_file_path: str = sample_data["audio_file_path"]
        # Get offset and duration
        offset: float = sample_data["offset"]
        duration: float = sample_data["duration"]
        # Load audio file
        waveform, sample_rate = lr.load(
            path=audio_file_path, sr=self.sample_rate, offset=offset, duration=duration
        )
        assert sample_rate == self.sample_rate
        if waveform.shape[0] != self.sample_rate * self.sample_seconds:
            raise ValueError(
                f"Sample at index {idx} has shape {waveform.shape}. "
                f"Expected shape is {(self.sample_rate * self.sample_seconds,)}"
            )
        # Normalize waveform to [-1, 1]
        if normalize:
            waveform = lr.util.normalize(waveform)
        return waveform, sample_rate

    def __len__(self) -> int:
        """Get number of samples in dataset.

        Returns:
            int: number of samples
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample and label at index idx.

        Args:
            idx (int): index of sample

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sample and label
        """
        # Get sample data
        sample_data: pd.Series = self.data.iloc[idx]
        # Load audio
        try:
            waveform, _ = self._load_audio(idx)
        except ValueError:
            warnings.warn(f"Sample at index {idx} could not be loaded.")
            # Try next item, check bounds
            if idx + 1 >= len(self):
                return self.__getitem__(idx - np.random.randint(1, 5))
            return self.__getitem__(idx + np.random.randint(1, min(5, len(self) - idx)))
        sample = torch.from_numpy(waveform).float()
        # Get label
        target = torch.tensor(sample_data["target"])
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


class ESC50SpectrogramDataset(ESC50WaveformDataset):
    """Dataset for ESC-50 dataset."""

    def __init__(
        self,
        annotations_file: str | pd.DataFrame,
        data_dir: str,
        sample_seconds: float = 1.0,
        sample_rate: int = 16000,
        filter_classes: str | int | list[str] | list[int] | None = None,
        filter_folds: int | list[int] | None = None,
        prune_invalid: bool = False,
        download: bool = False,
        verbose: bool = False,
        window_length_seconds: float = 0.025,
        hop_length_seconds: float = 0.01,
        num_mel_filters: int = 64,
        mel_filterbank_min_freq: float = 125.0,
        mel_filterbank_max_freq: float = 7500.0,
        log_offset: float = 0.01,
        example_window_length_seconds: float = 0.96,
        example_hop_length_seconds: float = 0.96,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            annotations_file (str | pd.DataFrame): path to annotations file or dataframe
            data_dir (str): path to audio files
            sample_seconds (float, optional): duration of samples in seconds.
                Defaults to 1.0.
            sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
            filter_classes (str | int | list[str] | list[int] | None, optional):
                classes to filter. Defaults to None.
            filter_folds (int | list[int] | None, optional): folds to filter.
                Defaults to None.
            prune_invalid (bool, optional): whether to prune invalid samples.
                Defaults to False.
            download (bool, optional): whether to download dataset. Defaults to False.
            verbose (bool, optional): whether to print progress. Defaults to True.
            window_length_seconds (float, optional): window length in seconds.
                Defaults to 0.025.
            hop_length_seconds (float, optional): hop length in seconds.
                Defaults to 0.01.
            num_mel_filters (int, optional): number of mel filters. Defaults to 64.
            mel_filterbank_min_freq (float, optional): minimum frequency for mel
                filterbank
            mel_filterbank_max_freq (float, optional): maximum frequency for mel
                filterbank
            log_offset (float, optional): offset for log-mel spectrogram.
                Defaults to 0.01.
            example_window_length_seconds (float, optional): window length for
                spectrogram examples in seconds. Defaults to 0.96.
            example_hop_length_seconds (float, optional): hop length for spectrogram
                examples in seconds. Defaults to 0.96.
            transform (Callable, optional): transform to apply to samples.
                Defaults to None.
            target_transform (Callable, optional): transform to apply to labels.
                Defaults to None.
        """
        super().__init__(
            annotations_file=annotations_file,
            data_dir=data_dir,
            sample_seconds=sample_seconds,
            sample_rate=sample_rate,
            filter_classes=filter_classes,
            filter_folds=filter_folds,
            prune_invalid=prune_invalid,
            download=download,
            verbose=verbose,
            transform=transform,
            target_transform=target_transform,
        )
        self.window_length_seconds: float = window_length_seconds
        self.hop_length_seconds: float = hop_length_seconds
        self.num_mel_filters: int = num_mel_filters
        self.mel_filterbank_min_freq: float = mel_filterbank_min_freq
        self.mel_filterbank_max_freq: float = mel_filterbank_max_freq
        self.log_offset: float = log_offset
        self.example_window_length_seconds: float = example_window_length_seconds
        self.example_hop_length_seconds: float = example_hop_length_seconds
        # Compute spectrogram parameters
        self.window_length_samples = int(
            np.round(self.window_length_seconds * self.sample_rate)
        )
        self.hop_length_samples = int(
            np.round(self.hop_length_seconds * self.sample_rate)
        )
        self.fft_length_samples = 2 ** int(
            np.ceil(np.log(self.window_length_samples) / np.log(2.0))
        )
        self.spectrogram_sample_rate = 1.0 / self.hop_length_seconds
        self.example_window_length_samples = int(
            np.round(self.example_window_length_seconds * self.spectrogram_sample_rate)
        )
        self.example_hop_length_samples = int(
            np.round(self.example_hop_length_seconds * self.spectrogram_sample_rate)
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample and label at index idx.

        Args:
            idx (int): index of sample

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sample and label
        """
        # Get sample data
        sample_data: pd.Series = self.data.iloc[idx]
        # Load audio
        try:
            waveform, _ = self._load_audio(idx)
        except ValueError:
            warnings.warn(f"Sample at index {idx} could not be loaded.")
            # Try next item, check bounds
            if idx + 1 >= len(self):
                return self.__getitem__(idx - np.random.randint(1, 5))
            return self.__getitem__(idx + np.random.randint(1, min(5, len(self) - idx)))
        # Compute log-mel spectrogram
        stft = lr.stft(
            y=waveform,
            n_fft=self.fft_length_samples,
            hop_length=self.hop_length_samples,
            win_length=self.window_length_samples,
        )
        mel_filterbank = lr.filters.mel(
            sr=self.sample_rate,
            n_fft=self.fft_length_samples,
            n_mels=self.num_mel_filters,
            fmin=self.mel_filterbank_min_freq,
            fmax=self.mel_filterbank_max_freq,
        )
        mel_spectrogram = np.dot(mel_filterbank, np.abs(stft))
        log_mel_spectrogram = np.log(mel_spectrogram + self.log_offset)
        # Frame spectrogram into sample
        log_mel_examples = lr.util.frame(
            log_mel_spectrogram,
            frame_length=self.example_window_length_samples,
            hop_length=self.example_hop_length_samples,
        )
        # Swap axes to make patches the first dimension
        log_mel_examples = np.transpose(log_mel_examples, axes=(2, 0, 1))
        sample = torch.from_numpy(log_mel_examples.copy()).float()
        sample = sample.squeeze()
        # Get label
        target = torch.tensor(sample_data["target"])
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


class ESC50MFCCDataset(ESC50SpectrogramDataset):
    """Dataset for ESC-50 dataset."""

    def __init__(
        self,
        annotations_file: str | pd.DataFrame,
        data_dir: str,
        sample_seconds: float = 1.0,
        sample_rate: int = 16000,
        filter_classes: str | int | list[str] | list[int] | None = None,
        filter_folds: int | list[int] | None = None,
        prune_invalid: bool = False,
        download: bool = False,
        verbose: bool = False,
        num_mfcc: int = 20,
        window_length_seconds: float = 0.025,
        hop_length_seconds: float = 0.01,
        num_mel_filters: int = 64,
        mel_filterbank_min_freq: float = 125.0,
        mel_filterbank_max_freq: float = 7500.0,
        log_offset: float = 0.01,
        example_window_length_seconds: float = 0.96,
        example_hop_length_seconds: float = 0.96,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        """Initialize dataset.

        Args:
            annotations_file (str | pd.DataFrame): path to annotations file or dataframe
            data_dir (str): path to audio files
            sample_seconds (float, optional): duration of samples in seconds.
                Defaults to 1.0.
            sample_rate (int, optional): sample rate of audio files. Defaults to 16000.
            filter_classes (str | int | list[str] | list[int] | None, optional):
                classes to filter. Defaults to None.
            filter_folds (int | list[int] | None, optional): folds to filter.
                Defaults to None.
            prune_invalid (bool, optional): whether to prune invalid samples.
                Defaults to False.
            download (bool, optional): whether to download dataset. Defaults to False.
            verbose (bool, optional): whether to print progress. Defaults to True.
            num_mfcc (int, optional): number of MFCCs. Defaults to 20.
            window_length_seconds (float, optional): window length in seconds.
                Defaults to 0.025.
            hop_length_seconds (float, optional): hop length in seconds.
                Defaults to 0.01.
            num_mel_filters (int, optional): number of mel filters. Defaults to 64.
            mel_filterbank_min_freq (float, optional): minimum frequency for mel
                filterbank. Defaults to 125.0.
            mel_filterbank_max_freq (float, optional): maximum frequency for mel
                filterbank. Defaults to 7500.0.
            log_offset (float, optional): offset for log-mel spectrogram.
                Defaults to 0.01.
            example_window_length_seconds (float, optional): window length for
                spectrogram examples in seconds. Defaults to 0.96.
            example_hop_length_seconds (float, optional): hop length for spectrogram
                examples in seconds. Defaults to 0.96.
            transform (Callable, optional): transform to apply to samples.
                Defaults to None.
            target_transform (Callable, optional): transform to apply to labels.
                Defaults to None.
        """
        super().__init__(
            annotations_file=annotations_file,
            data_dir=data_dir,
            sample_seconds=sample_seconds,
            sample_rate=sample_rate,
            filter_classes=filter_classes,
            filter_folds=filter_folds,
            prune_invalid=prune_invalid,
            download=download,
            verbose=verbose,
            window_length_seconds=window_length_seconds,
            hop_length_seconds=hop_length_seconds,
            num_mel_filters=num_mel_filters,
            mel_filterbank_min_freq=mel_filterbank_min_freq,
            mel_filterbank_max_freq=mel_filterbank_max_freq,
            log_offset=log_offset,
            example_window_length_seconds=example_window_length_seconds,
            example_hop_length_seconds=example_hop_length_seconds,
            transform=transform,
            target_transform=target_transform,
        )
        self.num_mfcc: int = num_mfcc

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample and label at index idx.

        Args:
            idx (int): index of sample

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sample and label
        """
        # Get sample data
        sample_data: pd.Series = self.data.iloc[idx]
        # Load audio
        try:
            waveform, _ = self._load_audio(idx)
        except ValueError:
            warnings.warn(f"Sample at index {idx} could not be loaded.")
            # Try next item, check bounds
            if idx + 1 >= len(self):
                return self.__getitem__(idx - np.random.randint(1, 5))
            return self.__getitem__(idx + np.random.randint(1, min(5, len(self) - idx)))
        # Compute MFCCs
        mfccs = lr.feature.mfcc(
            y=waveform,
            sr=self.sample_rate,
            n_mfcc=self.num_mfcc,
            n_fft=self.fft_length_samples,
            hop_length=self.hop_length_samples,
            win_length=self.window_length_samples,
            n_mels=self.num_mel_filters,
            fmin=self.mel_filterbank_min_freq,
            fmax=self.mel_filterbank_max_freq,
        )
        # Frame MFCCs into sample
        mfcc_examples = lr.util.frame(
            mfccs,
            frame_length=self.example_window_length_samples,
            hop_length=self.example_hop_length_samples,
        )
        # Swap axes to make patches the first dimension
        mfcc_examples = np.transpose(mfcc_examples, axes=(2, 0, 1))
        sample = torch.from_numpy(mfcc_examples.copy()).float()
        sample = sample.squeeze()
        # Get label
        target = torch.tensor(sample_data["target"])
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        if self.target_transform:
            target = self.target_transform(target)

        return sample, target


# %%
