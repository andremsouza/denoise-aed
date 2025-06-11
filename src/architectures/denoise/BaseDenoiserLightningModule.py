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
    ShortTimeObjectiveIntelligibility
)

class BaseDenoiserLightningModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        lr_scheduler_patience: int = 5,
        process_variance: float = 1e-5,
        initial_measurement_noise: float = 1e-4,
        adaptation_interval: int = 100,
        window_size: int = 1024, 
        hop_size: int = 512,
        noise_reduction_factor: float = 1.0,
        noise_window_duration: float = 0.1,
        sample_rate: int = 16000,
        alpha: float = 0.95,
        beta: float = 0.98,
        adaptive: bool = True
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_scheduler_patience = lr_scheduler_patience
        self.sample_rate = sample_rate
        self.process_variance = process_variance
        self.initial_measurement_noise = initial_measurement_noise
        self.adaptation_interval = adaptation_interval
        self.window_size = window_size
        self.hop_size = hop_size
        self.alpha = alpha
        self.beta = beta
        self.adaptive = adaptive
        self.noise_reduction_factor = noise_reduction_factor
        self.noise_window_duration = noise_window_duration

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
                self.sample_rate,
                extended=True),
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
            {
                k: v.compute()
                for k, v in self.metrics[stage].items()
            },
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

    def configure_optimizers(self) -> Any:
        """Configure optimizers.

        Returns:
            Optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.1,
            patience=self.lr_scheduler_patience,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "epoch", "monitor": "val_loss"}
        ]

    def _load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        """Load checkpoint if provided.

        Args:
            checkpoint_path: Path to checkpoint
        """
        if not checkpoint_path:
            return

        try:
            checkpoint = torch.load(checkpoint_path)
            # Assuming self.model is intended to be an nn.Module
            assert isinstance(
                self.model, torch.nn.Module
            ), "self.model is not an nn.Module instance"
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint)
            self.print(f"Successfully loaded checkpoint from {checkpoint_path}")
        # pylint: disable=W0718
        except Exception as e:
            self.print(f"WARNING: Failed to load checkpoint: {str(e)}")

    def on_train_epoch_end(self) -> None:
        """Actions to perform at the end of each training epoch.
        This method resets the metrics for the next epoch.
        """
        for metric in self.metrics["train"].values():
            metric.reset()
        torch.cuda.empty_cache()
        super().on_train_epoch_end()

    def on_validation_epoch_end(self) -> None:
        """Actions to perform at the end of each validation epoch.
        This method resets the metrics for the next epoch.
        """
        for metric in self.metrics["val"].values():
            metric.reset()
        torch.cuda.empty_cache()
        super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        """Actions to perform at the end of each test epoch.
        This method resets the metrics for the next epoch.
        """
        for metric in self.metrics["test"].values():
            metric.reset()
        torch.cuda.empty_cache()
        super().on_test_epoch_end()

    def on_predict_epoch_end(self) -> None:
        """Actions to perform at the end of each prediction epoch.
        This method resets the metrics for the next epoch.
        """
        torch.cuda.empty_cache()
        super().on_predict_epoch_end()
