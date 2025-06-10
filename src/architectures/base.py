"""Base Lightning Module for audio classification models."""

from typing import (
    Any,
    Literal,
)

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelHammingDistance,
    MultilabelPrecision,
    MultilabelRecall,
    # MultilabelROC,
)

# For batch data (inputs, targets)
Batch = tuple[torch.Tensor, torch.Tensor]

# For metric dictionaries
MetricsDict = dict[str, Any]


class BaseAudioLightningModule(pl.LightningModule):
    """Base class for audio classification Lightning modules."""

    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        metrics_threshold: float = 0.5,
        lr_scheduler_patience: int = 5,
        num_classes: int = 527,
    ) -> None:
        """Initialize the base module.

        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            metrics_threshold: Threshold for metrics calculation
            lr_scheduler_patience: Patience for learning rate scheduler
            num_classes: Number of classes for classification
        """
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.metrics_threshold = metrics_threshold
        self.lr_scheduler_patience = lr_scheduler_patience
        self.num_classes = num_classes

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
            f"{prefix}_acc": MultilabelAccuracy(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="weighted",
            ),
            f"{prefix}_auroc": MultilabelAUROC(
                num_labels=self.num_classes, average="weighted"
            ),
            f"{prefix}_ap": MultilabelAveragePrecision(
                num_labels=self.num_classes, average="weighted"
            ),
            f"{prefix}_f1": MultilabelF1Score(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="weighted",
            ),
            f"{prefix}_hamming": MultilabelHammingDistance(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="weighted",
            ),
            f"{prefix}_precision": MultilabelPrecision(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="weighted",
            ),
            f"{prefix}_recall": MultilabelRecall(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="weighted",
            ),
            # f"{prefix}_roc": MultilabelROC(num_labels=self.num_classes),
            f"{prefix}_acc_per_class": MultilabelAccuracy(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="none",
            ),
            f"{prefix}_auroc_per_class": MultilabelAUROC(
                num_labels=self.num_classes, average="none"
            ),
            f"{prefix}_ap_per_class": MultilabelAveragePrecision(
                num_labels=self.num_classes, average="none"
            ),
            f"{prefix}_f1_per_class": MultilabelF1Score(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="none",
            ),
            f"{prefix}_hamming_per_class": MultilabelHammingDistance(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="none",
            ),
            f"{prefix}_precision_per_class": MultilabelPrecision(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="none",
            ),
            f"{prefix}_recall_per_class": MultilabelRecall(
                num_labels=self.num_classes,
                threshold=self.metrics_threshold,
                average="none",
            ),
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
        self, stage: str, targets: torch.Tensor, outputs: torch.Tensor
    ) -> None:
        """Calculate metrics for the given stage.

        Args:
            stage: Stage name (train, val, test)
            targets: Target labels
            outputs: Model outputs
        """
        for metric in self.metrics[stage].values():
            metric(outputs.to(metric.device), targets.int().to(metric.device))

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
                if not k.endswith("_per_class")
            },
            prog_bar=True,
        )
        for i in range(self.num_classes):
            self.log_dict(
                {
                    f"{k}_{i}": v.compute()[i]
                    for k, v in self.metrics[stage].items()
                    if k.endswith("_per_class")
                },
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
        inputs, targets = batch
        outputs = self(inputs)
        loss = F.binary_cross_entropy(outputs, targets)
        self._calculate_metrics(stage, targets, outputs)
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
