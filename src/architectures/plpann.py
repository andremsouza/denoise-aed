import warnings

import torch

from src.architectures.base import BaseAudioLightningModule


class PLPANN(BaseAudioLightningModule):
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
        super().__init__(
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            metrics_threshold=metrics_threshold,
            lr_scheduler_patience=lr_scheduler_patience,
            num_classes=num_classes,
        )

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from the model."""
        return self.model(x)["embedding"]

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass."""
        # pylint: disable=arguments-differ
        return self.model(x)["clipwise_output"]

    def _load_checkpoint(self, checkpoint_path: str | None = None) -> None:
        """Load checkpoint if provided.

        Args:
            checkpoint_path: Path to checkpoint
        """
        if not checkpoint_path:
            return

        try:
            checkpoint = torch.load(checkpoint_path)
            if "state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["state_dict"])
            else:
                self.model.load_state_dict(checkpoint["model"])
        except (
            FileNotFoundError,
            RuntimeError,
            torch.serialization.pickle.UnpicklingError,
        ) as e:
            warnings.warn(f"Failed to load checkpoint from {checkpoint_path}: {e}")

    def _load_pretrained_weights(
        self, checkpoint_path: str | None = None, num_classes_pretrained: int = 527
    ) -> None:
        """Load pretrained weights if provided.
        If num_classes in the pretrained model is different from the current model,
        the last layer will be replaced with a new one.
        This is useful for transfer learning.

        Args:
            checkpoint_path: Path to checkpoint
            num_classes_pretrained: Number of classes in the pretrained model
        """
        if not checkpoint_path:
            return
        if num_classes_pretrained != self.num_classes:
            # Alter model to match num_classes_pretrained
            self.model.fc_audioset = torch.nn.Linear(
                self.model.fc_audioset.in_features, num_classes_pretrained, bias=True
            )
        self._load_checkpoint(checkpoint_path)
        if self.model.fc_audioset.out_features != self.num_classes:
            # Alter model to match num_classes
            self.model.fc_audioset = torch.nn.Linear(
                self.model.fc_audioset.in_features, self.num_classes, bias=True
            )
