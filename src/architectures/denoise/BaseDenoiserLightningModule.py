"""Base Lightning Module for audio classification models."""

from typing import (
    Any,
    Literal,
)

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from torchmetrics.audio import (
    SignalNoiseRatio,
    SignalDistortionRatio,
    ShortTimeObjectiveIntelligibility,
)

from src.config.denoiser_config import DenoiserConfig


class BaseDenoiserLightningModule(pl.LightningModule):
    def __init__(self, config: DenoiserConfig) -> None:
        super().__init__()
        self.sample_rate = config.sample_rate
        self.process_variance = config.process_variance
        self.initial_measurement_noise = config.initial_measurement_noise
        self.adaptation_interval = config.adaptation_interval
        self.window_size = config.window_size
        self.hop_size = config.hop_size
        self.alpha = config.alpha
        self.beta = config.beta
        self.adaptive = config.adaptive
        self.noise_reduction_factor = config.noise_reduction_factor
        self.noise_window_duration = config.noise_window_duration

        # Metrics will be initialized in child classes after model creation
        self.metrics: dict[str, dict[str, Any]] = {}

    def _init_metrics(self, prefix: str) -> dict[str, Any]:
        """Initialize metrics for a specific stage.

        Args:
            prefix: Stage prefix (train, val, test)

        Returns:
            dict[str, Any]: Dictionary of metrics
        """
        metrics = {
            f"{prefix}_snr": SignalNoiseRatio(),
            f"{prefix}_stoi": ShortTimeObjectiveIntelligibility(
                self.sample_rate, extended=True
            ),
            f"{prefix}_sdr": SignalDistortionRatio(load_diag=1e-6),
        }
        return metrics

    def setup_metrics(self) -> None:
        """Set up metrics for all stages."""
        self.metrics = {
            "train": self._init_metrics(prefix="train"),
            "val": self._init_metrics(prefix="val"),
            "test": self._init_metrics(prefix="test"),
        }

    def _calculate_metrics(
        self, stage: str, inputs: torch.Tensor, outputs: torch.Tensor
    ) -> None:
        for metric in self.metrics[stage].values():
            metric(outputs.to(metric.device), inputs.to(metric.device))

    def _log_metrics(self, stage: str) -> None:
        """Log metrics for the given stage.

        Args:
            stage: Stage name (train, val, test)
        """
        if stage == "train":
            self.log(
                "lr",
                self.trainer.optimizers[0].param_groups[0]["lr"],
                prog_bar=True,
            )
        self.log_dict(
            {k: v.compute() for k, v in self.metrics[stage].items()},
            prog_bar=True,
        )

    def shared_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        stage: Literal["train", "val", "test"],
    ) -> torch.Tensor:
        """Shared step for training, validation and testing.

        Args:
            batch: Batch of data (inputs, targets)
            stage: Stage name

        Returns:
            Loss tensor
        """
        inputs, _ = batch
        outputs = self(inputs)
        loss = F.l1_loss(outputs, inputs)
        self._calculate_metrics(stage, inputs, outputs)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self._log_metrics(stage)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, *args, **kwargs
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Batch of data
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        # pylint: disable=W0221, W0613
        return self.shared_step(batch, "train")

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        dataloader_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Validation step.

        Args:
            batch: Batch of data
            batch_idx: Batch index
            dataloader_idx: Dataloader index

        Returns:
            Loss tensor
        """
        # pylint: disable=W0221, W0613
        return self.shared_step(batch, "val")

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        *args,
        dataloader_idx: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Test step.

        Args:
            batch: Batch of data
            batch_idx: Batch index
            dataloader_idx: Dataloader index

        Returns:
            Loss tensor
        """
        # pylint: disable=W0221, W0613
        return self.shared_step(batch, "test")

    def predict_step(
        self,
        batch: list[torch.Tensor],
        batch_idx: int,
        *args,
        dataloader_idx: int = 0,
        **kwargs,
    ) -> Any:
        """Predict step.

        Args:
            batch: Batch of data
            batch_idx: Batch index
            dataloader_idx: Dataloader index

        Returns:
            Prediction outputs
        """
        # pylint: disable=W0221, W0613
        inputs = batch[0]
        outputs = self(inputs)
        return outputs
