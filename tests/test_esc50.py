"""This module contains unit tests for the ESC-50 dataset classes."""

import os
import sys

import pytest
import pandas as pd
import torch

from src.datasets.esc50 import (
    ESC50WaveformDataset,
    ESC50SpectrogramDataset,
    ESC50MFCCDataset,
)

# Constants for testing
TEST_DATA_DIR = "tests/data/ESC-50/audio"
TEST_META_FILE = "tests/data/ESC-50/meta/esc50.csv"
SAMPLE_RATE = 16000
SAMPLE_SECONDS = 1.0


# Fixtures
@pytest.fixture
def esc50_annotations():
    """Fixture for loading a sample ESC-50 annotations DataFrame."""
    data = {
        "filename": ["1-100032-A-0.wav", "1-100038-A-14.wav"],
        "fold": [1, 1],
        "target": [0, 14],
        "category": ["dog", "chirping_birds"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def esc50_waveform_dataset(esc50_annotations):
    """Fixture for initializing ESC50WaveformDataset."""
    return ESC50WaveformDataset(
        annotations_file=esc50_annotations,
        data_dir=TEST_DATA_DIR,
        sample_seconds=SAMPLE_SECONDS,
        sample_rate=SAMPLE_RATE,
        prune_invalid=False,
        download=False,
        verbose=False,
    )


@pytest.fixture
def esc50_spectrogram_dataset(esc50_annotations):
    """Fixture for initializing ESC50SpectrogramDataset."""
    return ESC50SpectrogramDataset(
        annotations_file=esc50_annotations,
        data_dir=TEST_DATA_DIR,
        sample_seconds=SAMPLE_SECONDS,
        sample_rate=SAMPLE_RATE,
        prune_invalid=False,
        download=False,
        verbose=False,
    )


@pytest.fixture
def esc50_mfcc_dataset(esc50_annotations):
    """Fixture for initializing ESC50MFCCDataset."""
    return ESC50MFCCDataset(
        annotations_file=esc50_annotations,
        data_dir=TEST_DATA_DIR,
        sample_seconds=SAMPLE_SECONDS,
        sample_rate=SAMPLE_RATE,
        prune_invalid=False,
        download=False,
        verbose=False,
    )


# Tests
def test_waveform_dataset_initialization(esc50_waveform_dataset):
    """Test initialization of ESC50WaveformDataset."""
    assert len(esc50_waveform_dataset) > 0
    assert esc50_waveform_dataset.sample_rate == SAMPLE_RATE
    assert esc50_waveform_dataset.sample_seconds == SAMPLE_SECONDS


def test_spectrogram_dataset_initialization(esc50_spectrogram_dataset):
    """Test initialization of ESC50SpectrogramDataset."""
    assert len(esc50_spectrogram_dataset) > 0
    assert esc50_spectrogram_dataset.num_mel_filters == 64


def test_mfcc_dataset_initialization(esc50_mfcc_dataset):
    """Test initialization of ESC50MFCCDataset."""
    assert len(esc50_mfcc_dataset) > 0
    assert esc50_mfcc_dataset.num_mfcc == 20


def test_waveform_dataset_getitem(esc50_waveform_dataset):
    """Test __getitem__ for ESC50WaveformDataset."""
    sample, label = esc50_waveform_dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert sample.shape[0] == SAMPLE_RATE * SAMPLE_SECONDS


def test_spectrogram_dataset_getitem(esc50_spectrogram_dataset):
    """Test __getitem__ for ESC50SpectrogramDataset."""
    sample, label = esc50_spectrogram_dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert len(sample.shape) == 2  # Spectrogram shape


def test_mfcc_dataset_getitem(esc50_mfcc_dataset):
    """Test __getitem__ for ESC50MFCCDataset."""
    sample, label = esc50_mfcc_dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert isinstance(label, torch.Tensor)
    assert len(sample.shape) == 2  # MFCC shape


def test_filter_classes(esc50_waveform_dataset):
    """Test filtering by classes."""
    esc50_waveform_dataset.filter_classes = [0]
    esc50_waveform_dataset._filter_classes()
    assert all(esc50_waveform_dataset.annotations["target"] == 0)


def test_filter_folds(esc50_waveform_dataset):
    """Test filtering by folds."""
    esc50_waveform_dataset.filter_folds = [1]
    esc50_waveform_dataset._filter_folds()
    assert all(esc50_waveform_dataset.annotations["fold"] == 1)


def test_prune_invalid_samples(esc50_waveform_dataset):
    """Test pruning invalid samples."""
    esc50_waveform_dataset.prune_invalid = True
    esc50_waveform_dataset._prune_invalid_samples()
    assert len(esc50_waveform_dataset) > 0


@pytest.mark.skip(
    reason="Skipping test_download_dataset as it requires internet access."
)
def test_download_dataset():
    """Test downloading the dataset."""
    _ = ESC50WaveformDataset(
        annotations_file=TEST_META_FILE,
        data_dir=TEST_DATA_DIR,
        download=True,
        verbose=True,
    )
    assert os.path.exists(TEST_DATA_DIR)
