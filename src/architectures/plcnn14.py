"""PyTorch Lightning Module for Cnn14 (PANNs)."""

import torch

from src.architectures.plpann import PLPANN
from src.architectures.external.panns import Cnn14
from src.config.model_config import PLCnn14Config


class PLCnn14(PLPANN):
    """PyTorch Lightning Module for Cnn14 (PANNs)."""

    def __init__(self, config: PLCnn14Config) -> None:
        """Initialize the model with a configuration object.

        Args:
            config: Configuration for the model
        """
        super().__init__(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            metrics_threshold=config.metrics_threshold,
            lr_scheduler_patience=config.lr_scheduler_patience,
            num_classes=config.num_classes,
        )

        self.model = Cnn14(
            sample_rate=config.sample_rate,
            window_size=config.window_size,
            hop_size=config.hop_size,
            mel_bins=config.mel_bins,
            fmin=config.fmin,
            fmax=config.fmax,
            classes_num=config.num_classes,
        )
        self.checkpoint_path = config.checkpoint_path
        self._load_pretrained_weights(self.checkpoint_path, num_classes_pretrained=527)

        # Initialize metrics
        self.setup_metrics()
