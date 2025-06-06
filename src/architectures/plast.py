"""Adaptation of the ASTModel for PyTorch Lightning."""

# %% [markdown]
# # Imports

# %%
import torch
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from src.architectures.base import BaseAudioLightningModule
from src.architectures.external.ast_models import ASTModel
from src.config.model_config import PLASTConfig

# %% [markdown]
# # Classes

# %%


class PLAST(BaseAudioLightningModule):
    """Adaptation of the ASTModel for PyTorch Lightning."""

    def __init__(self, config: PLASTConfig) -> None:
        """Initialize AST model.

        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            metrics_threshold: Threshold for metrics calculation
            lr_scheduler_patience: Patience for learning rate scheduler
            label_dim: Number of labels
            fstride: Stride of frequency dimension
            tstride: Stride of time dimension
            input_fdim: Frequency dimension of input
            input_tdim: Time dimension of input
            imagenet_pretrain: Whether to use imagenet pretrained weights
            audioset_pretrain: Whether to use audioset pretrained weights
            model_size: Model size
            verbose: Whether to print model information
            checkpoint_path: Path to checkpoint
        """
        super().__init__(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            metrics_threshold=config.metrics_threshold,
            lr_scheduler_patience=config.lr_scheduler_patience,
            num_classes=config.num_classes,
        )

        # Initialize AST model
        self.model = ASTModel(
            label_dim=config.num_classes,
            fstride=config.fstride,
            tstride=config.tstride,
            input_fdim=config.input_fdim,
            input_tdim=config.input_tdim,
            imagenet_pretrain=config.imagenet_pretrain,
            audioset_pretrain=config.audioset_pretrain,
            model_size=config.model_size,
            verbose=config.verbose,
        )

        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None
        # Spectrogram extractor
        self.model.spectrogram_extractor = Spectrogram(
            n_fft=config.window_size,
            hop_length=config.hop_size,
            win_length=config.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        # Logmel feature extractor
        self.model.logmel_extractor = LogmelFilterBank(
            sr=config.sample_rate,
            n_fft=config.window_size,
            n_mels=config.mel_bins,
            fmin=config.fmin,
            fmax=config.fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )
        # Spec augmenter
        self.model.spec_augmenter = SpecAugmentation(
            time_drop_width=config.input_tdim,
            time_stripes_num=2,
            freq_drop_width=config.input_fdim,
            freq_stripes_num=2,
        )

        self.checkpoint_path = config.checkpoint_path
        self._load_checkpoint(self.checkpoint_path)

        # Initialize metrics
        self.setup_metrics()

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from the model.

        Args:
            x: Input tensor

        Returns:
            Embedding tensor
        """
        # expect input x = (batch_size, time_frame_num, frequency_bins), e.g., (12, 1024, 128)
        # pylint: disable=W0221
        # (batch_size, data_length)
        x = self.model.spectrogram_extractor(x)
        # (batch_size, 1, time_steps, freq_bins)
        x = self.model.logmel_extractor(x)
        # (batch_size, 1, time_steps, mel_bins)
        if self.training:
            x = self.model.spec_augmenter(x)
        # x = x.unsqueeze(1)
        x = x.transpose(2, 3)

        batch_size = x.shape[0]
        x = self.model.v.patch_embed(x)
        cls_tokens = self.model.v.cls_token.expand(batch_size, -1, -1)
        dist_token = self.model.v.dist_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + self.model.v.pos_embed
        x = self.model.v.pos_drop(x)
        for blk in self.model.v.blocks:
            x = blk(x)
        x = self.model.v.norm(x)
        x = (x[:, 0] + x[:, 1]) / 2

        return x

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Forward pass through the model and extract embeddings
        x = self.extract_embeddings(x)
        # Apply the classification head
        x = self.model.mlp_head(x)
        # Return sigmoid output for compatibility with base metrics
        return torch.sigmoid(x)


# %%
