import lightning.pytorch as pl
import torch
import torch.nn as nn
import pytest
from torch.utils.data import TensorDataset, DataLoader

from src.architectures.plast import PLAST
from src.config.model_config import PLASTConfig


@pytest.fixture
def dummy_config():
    # Create a minimal PLASTConfig tailored for testing.
    return PLASTConfig(label_dim=5)


@pytest.fixture
def model(dummy_config):
    # Instantiate the PLAST model.
    model = PLAST(dummy_config)
    return model


def test_forward(model):
    batch_size = 4
    x = torch.randn(batch_size, 16000)
    x = x.to(model.device)
    output = model(x)
    # The forward call applies mlp_head so the output should be (batch, 5)
    assert output.shape == (batch_size, 5)


def test_extract_embeddings(model):
    batch_size = 4
    x = torch.randn(batch_size, 16000)
    x = x.to(model.device)
    embeddings = model.extract_embeddings(x)
    # resulting in a tensor of shape (batch, 768)
    assert embeddings.shape == (batch_size, 768)


def test_trainer_prediction(model):

    # Create a small dummy dataset
    batch_size = 4
    x = torch.randn(batch_size, 16000)
    x = x.to(model.device)

    # Create a simple DataLoader
    dataset = TensorDataset(x)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=0,
    )

    # Initialize a Trainer with minimal configuration
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
    )

    # Run prediction
    predictions = trainer.predict(model, dataloaders=dataloader)

    # Verify predictions
    assert len(predictions) == 1  # Should be a single batch
    assert predictions[0].shape == (batch_size, 5)  # Same output shape as forward
    assert isinstance(predictions[0], torch.Tensor)
    assert torch.all((predictions[0] >= 0) & (predictions[0] <= 1))  # Sigmoid outputs


def test_trainer_validation(model):
    # Create a small dummy dataset with inputs and targets
    batch_size = 4
    x = torch.randn(batch_size, 96, 64)  # Input features
    y = torch.randint(
        0, 2, (batch_size, 5), dtype=torch.float
    )  # Binary targets for 5 classes

    # Create a simple DataLoader with both inputs and targets
    dataset = TensorDataset(x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    # Initialize a Trainer with minimal configuration
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
    )

    # Run validation
    validation_results = trainer.validate(model, dataloaders=dataloader)

    # Verify validation results
    assert isinstance(validation_results, list)
    assert len(validation_results) == 1  # One result per validation run
    assert isinstance(
        validation_results[0], dict
    )  # Results should be a dictionary of metrics

    # Check for expected metrics in the results
    assert "val_loss" in validation_results[0]

    # Optional: Verify additional metrics if your model calculates them
    # assert "val_accuracy" in validation_results[0]

    # Check that the loss value is reasonable
    assert isinstance(validation_results[0]["val_loss"], float)
    assert validation_results[0]["val_loss"] >= 0  # Loss should be non-negative
