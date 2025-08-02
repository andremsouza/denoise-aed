"""This module contains unit tests for the Swine dataset classes."""

import sys
import os

import pytest
import pandas as pd
import numpy as np
import torch

from src.datasets.swine import (
    SwineWaveformDataset,
    SwineSpectrogramDataset,
    SwineMFCCDataset,
)


@pytest.fixture
def dummy_annotations():
    """
    Create a dummy pandas DataFrame containing annotation data.

    The DataFrame includes the following columns:
    - "duration_s": Duration in seconds for each annotation.
    - "class_1" to "class_9": Binary indicators (0 or 1) for different classes.

    The index of the DataFrame represents timestamps in the format "DD/MM/YYYY HH:MM:SS".

    Returns:
        pd.DataFrame: A DataFrame with dummy annotation data.
    """
    data = {
        "duration_s": [2.0, 3.0],
        "class_1": [1, 0],
        "class_2": [0, 1],
        "class_3": [0, 0],
        "class_4": [0, 0],
        "class_5": [0, 0],
        "class_6": [0, 0],
        "class_7": [0, 0],
        "class_8": [0, 0],
        "class_9": [0, 0],
    }
    df = pd.DataFrame(data, index=["30/09/2020 01:02:03", "30/09/2020 01:02:05"])
    return df


@pytest.fixture
def mock_listdir():
    """
    Mock function that simulates the behavior of os.listdir by returning a predefined list of
        filenames.

    Returns:
        list: A list of strings representing filenames.
    """
    return ["ALA_1_2_2020-09-30_01-02-03.wav", "ALA_1_3_2020-09-30_01-02-04.wav"]


@pytest.fixture
def mock_librosa_load():
    """
    Creates a mock implementation of the `librosa.load` function for testing purposes.

    Returns:
        function: A mock function that simulates loading an audio file. The mock function
        accepts the following parameters:
            - path (str, optional): The file path to the audio file. Defaults to None.
            - sr (int, optional): The sampling rate of the audio. Defaults to 16000.
            - offset (float, optional): The starting point in seconds to load the audio.
                Defaults to 0.0.
            - duration (float, optional): The duration in seconds of the audio to load.
                Defaults to 1.0.

        The mock function returns:
            - samples (numpy.ndarray): A NumPy array of random samples simulating audio data.
            - sr (int): The sampling rate of the audio.
    """

    def _mock_load(path=None, sr=16000, offset=0.0, duration=1.0):
        samples = np.random.randn(int(sr * duration))
        return samples, sr

    return _mock_load


@pytest.mark.parametrize(
    "dataset_cls", [SwineWaveformDataset, SwineSpectrogramDataset, SwineMFCCDataset]
)
def test_dataset_len(
    dataset_cls,
    dummy_annotations,
    mock_listdir,
    mock_librosa_load,
    tmp_path,
    monkeypatch,
):
    """Test the __len__ method of the dataset for multiple dataset classes."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    for fname in mock_listdir:
        (data_dir / fname).touch()
    monkeypatch.setattr(os, "listdir", lambda _: mock_listdir)
    monkeypatch.setattr("librosa.load", mock_librosa_load)
    ds = dataset_cls(annotations_file=dummy_annotations, data_dir=str(data_dir))
    assert len(ds) == int(sum(dummy_annotations["duration_s"]))


@pytest.mark.parametrize(
    "dataset_cls", [SwineWaveformDataset, SwineSpectrogramDataset, SwineMFCCDataset]
)
def test_dataset_iter(
    dataset_cls,
    dummy_annotations,
    mock_listdir,
    mock_librosa_load,
    tmp_path,
    monkeypatch,
):
    """Test the __iter__ method of the dataset for multiple dataset classes."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    for fname in mock_listdir:
        (data_dir / fname).touch()
    monkeypatch.setattr(os, "listdir", lambda _: mock_listdir)
    monkeypatch.setattr("librosa.load", mock_librosa_load)
    ds = dataset_cls(annotations_file=dummy_annotations, data_dir=str(data_dir))
    for sample, target in ds:
        assert isinstance(sample, torch.Tensor)
        assert isinstance(target, torch.Tensor)


@pytest.mark.parametrize(
    "dataset_cls", [SwineWaveformDataset, SwineSpectrogramDataset, SwineMFCCDataset]
)
def test_dataset_invalid_sample_rate(
    dataset_cls, dummy_annotations, mock_listdir, tmp_path, monkeypatch
):
    """Test behavior when audio files have an invalid sample rate."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    for fname in mock_listdir:
        (data_dir / fname).touch()

    def mock_load_invalid_sample_rate(*args, **kwargs):
        return np.random.randn(16000), 8000  # Invalid sample rate

    monkeypatch.setattr(os, "listdir", lambda _: mock_listdir)
    monkeypatch.setattr("librosa.load", mock_load_invalid_sample_rate)
    ds = dataset_cls(annotations_file=dummy_annotations, data_dir=str(data_dir))
    with pytest.raises(AssertionError):
        ds[0]


@pytest.mark.parametrize(
    "dataset_cls", [SwineWaveformDataset, SwineSpectrogramDataset, SwineMFCCDataset]
)
def test_dataset_no_audio_files(dataset_cls, dummy_annotations, tmp_path, monkeypatch):
    """Test behavior when no audio files are present in the directory."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    monkeypatch.setattr(os, "listdir", lambda _: [])
    ds = dataset_cls(annotations_file=dummy_annotations, data_dir=str(data_dir))
    assert len(ds) == 0


@pytest.mark.parametrize(
    "dataset_cls", [SwineWaveformDataset, SwineSpectrogramDataset, SwineMFCCDataset]
)
def test_dataset_transform_and_target_transform(
    dataset_cls,
    dummy_annotations,
    mock_listdir,
    mock_librosa_load,
    tmp_path,
    monkeypatch,
):
    """Test that both transforms and target transforms are applied correctly."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    for fname in mock_listdir:
        (data_dir / fname).touch()
    monkeypatch.setattr(os, "listdir", lambda _: mock_listdir)
    monkeypatch.setattr("librosa.load", mock_librosa_load)

    def mock_transform(x):
        # Mock transform that concatenates the waveform with itself, doubling its length
        return torch.cat((x, x), dim=0)

    def mock_target_transform(y):
        # Mock target transform that adds a new class
        return torch.cat((y, torch.tensor([1])), dim=0)

    ds = dataset_cls(
        annotations_file=dummy_annotations,
        data_dir=str(data_dir),
        transform=mock_transform,
        target_transform=mock_target_transform,
    )
    sample, target = ds[0]
    if dataset_cls == SwineWaveformDataset:
        assert sample.shape[0] == torch.tensor(mock_librosa_load()[0]).shape[0] * 2
    elif dataset_cls == SwineSpectrogramDataset:
        assert sample.shape[0] == ds.num_mel_filters * 2
    elif dataset_cls == SwineMFCCDataset:
        assert sample.shape[0] == ds.num_mfcc * 2
    # Check that the target has an additional class
    assert target.shape[0] == ds.num_classes + 1


def test_dataset_empty_annotations(tmp_path, monkeypatch):
    """Test behavior when annotations are empty."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    monkeypatch.setattr(os, "listdir", lambda _: [])
    empty_annotations = pd.DataFrame()
    ds = SwineWaveformDataset(
        annotations_file=empty_annotations, data_dir=str(data_dir)
    )
    assert len(ds) == 0


def test_dataset_invalid_audio_file(dummy_annotations, tmp_path, monkeypatch):
    """Test behavior when audio files are invalid."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    invalid_file = "invalid_audio.wav"
    (data_dir / invalid_file).touch()

    def mock_load_fail(*args, **kwargs):
        raise ValueError("Invalid audio file")

    monkeypatch.setattr(os, "listdir", lambda _: [invalid_file])
    monkeypatch.setattr("librosa.load", mock_load_fail)
    ds = SwineWaveformDataset(
        annotations_file=dummy_annotations, data_dir=str(data_dir)
    )
    assert len(ds) == 0


@pytest.mark.parametrize(
    "invalid_filename", ["bad_filename.wav", "ALA_X_2021-13-01_25-61-61.wav"]
)
def test_extract_data_from_filename_invalid(
    invalid_filename, dummy_annotations, tmp_path, monkeypatch
):
    """Check that a warning is issued for invalid filenames."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    (data_dir / invalid_filename).touch()
    monkeypatch.setattr(os, "listdir", lambda _: [invalid_filename])
    ds = SwineWaveformDataset(
        annotations_file=dummy_annotations,
        data_dir=str(data_dir),
    )
    assert len(ds) == 0


def test_dataset_prune_invalid_samples(
    dummy_annotations, mock_listdir, tmp_path, monkeypatch
):
    """Check that invalid samples are pruned when prune_invalid=True."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    for fname in mock_listdir:
        (data_dir / fname).touch()

    # Force a ValueError by mocking librosa.load to return a shorter waveform
    def mock_load_short(*args, **kwargs):
        return np.random.randn(8000), 16000  # half the expected length

    monkeypatch.setattr(os, "listdir", lambda _: mock_listdir)
    monkeypatch.setattr("librosa.load", mock_load_short)
    ds = SwineWaveformDataset(
        annotations_file=dummy_annotations,
        data_dir=str(data_dir),
        prune_invalid=True,
    )
    # Dataset should prune invalid samples
    assert len(ds) == 0


def test_dataset_load_audio_fallback(
    dummy_annotations, mock_listdir, tmp_path, monkeypatch
):
    """Check that dataset tries fallback index if load_audio fails."""
    data_dir = tmp_path / "audio"
    data_dir.mkdir()
    for fname in mock_listdir:
        (data_dir / fname).touch()
    # Mock librosa.load to raise ValueError for first call, succeed on second
    call_count = {"val": 0}

    def mock_load_fail_once(*args, **kwargs):
        if call_count["val"] == 0:
            call_count["val"] += 1
            raise ValueError("Test error")
        return np.random.randn(16000), 16000

    monkeypatch.setattr(os, "listdir", lambda _: mock_listdir)
    monkeypatch.setattr("librosa.load", mock_load_fail_once)
    ds = SwineWaveformDataset(
        annotations_file=dummy_annotations, data_dir=str(data_dir)
    )
    sample, target = ds[0]
    assert sample.shape[0] == 16000
    assert target.shape[0] == ds.num_classes
