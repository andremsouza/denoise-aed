"""Model configuration classes."""

from dataclasses import dataclass


@dataclass
class AudioModelConfig(object):
    """Base configuration for audio models.

    Args:
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        metrics_threshold: Threshold for metrics calculation
        lr_scheduler_patience: Patience for learning rate scheduler
        num_classes: Number of classes for classification
        checkpoint_path: Path to checkpoint
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    metrics_threshold: float = 0.5
    lr_scheduler_patience: int = 2
    sample_rate: int = 16000
    # window_size = 0.025s * 16000Hz = 400
    window_size: int = 1024
    # hop_size = 0.01s * 16000Hz = 160
    hop_size: int = 320
    mel_bins: int = 64
    fmin: int = 50
    fmax: int = 14000
    num_classes: int = 7
    checkpoint_path: str | None = None


@dataclass
class PLASTConfig(AudioModelConfig):
    """Configuration for AST models.

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

    label_dim: int = 7
    fstride: int = 10
    tstride: int = 10
    input_fdim: int = 64
    input_tdim: int = 51
    imagenet_pretrain: bool = True
    audioset_pretrain: bool = True
    model_size: str = "base384"
    verbose: bool = True
    checkpoint_path: str | None = None


@dataclass
class PLCnn14Config(AudioModelConfig):
    """Configuration for Cnn14 models.

    Args:
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        metrics_threshold: Threshold for metrics calculation
        lr_scheduler_patience: Patience for learning rate scheduler
        sample_rate: Sample rate of audio
        window_size: Window size for spectrogram calculation
        hop_size: Hop size for spectrogram calculation
        mel_bins: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        classes_num: Number of classes
        checkpoint_path: Path to checkpoint
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    metrics_threshold: float = 0.5
    checkpoint_path: str | None = "pretrained_models/Cnn14_mAP=0.431.pth"


@dataclass
class PLDaiNet19Config(AudioModelConfig):
    """Configuration for DaiNet19 models.

    Args:
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        metrics_threshold: Threshold for metrics calculation
        lr_scheduler_patience: Patience for learning rate scheduler
        sample_rate: Sample rate of audio
        window_size: Window size for spectrogram calculation
        hop_size: Hop size for spectrogram calculation
        mel_bins: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        classes_num: Number of classes
        checkpoint_path: Path to checkpoint
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    metrics_threshold: float = 0.5
    checkpoint_path: str | None = "pretrained_models/DaiNet19_mAP=0.295.pth"


@dataclass
class PLLeeNet24Config(AudioModelConfig):
    """Configuration for LeeNet24 models.

    Args:
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        metrics_threshold: Threshold for metrics calculation
        lr_scheduler_patience: Patience for learning rate scheduler
        sample_rate: Sample rate of audio
        window_size: Window size for spectrogram calculation
        hop_size: Hop size for spectrogram calculation
        mel_bins: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        classes_num: Number of classes
        checkpoint_path: Path to checkpoint
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    metrics_threshold: float = 0.5
    checkpoint_path: str | None = "pretrained_models/LeeNet24_mAP=0.336.pth"


@dataclass
class PLMobileNetV1Config(AudioModelConfig):
    """Configuration for MobileNetV1 models.

    Args:
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        metrics_threshold: Threshold for metrics calculation
        lr_scheduler_patience: Patience for learning rate scheduler
        sample_rate: Sample rate of audio
        window_size: Window size for spectrogram calculation
        hop_size: Hop size for spectrogram calculation
        mel_bins: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        classes_num: Number of classes
        checkpoint_path: Path to checkpoint
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    metrics_threshold: float = 0.5
    checkpoint_path: str | None = "pretrained_models/MobileNetV1_mAP=0.389.pth"


@dataclass
class PLMobileNetV2Config(AudioModelConfig):
    """Configuration for MobileNetV2 models.

    Args:
        sample_rate: Sample rate for audio
        window_size: Window size for STFT
        hop_size: Hop size for STFT
        mel_bins: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        classes_num: Number of classes
        checkpoint_path: Path to checkpoint
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    metrics_threshold: float = 0.5
    checkpoint_path: str | None = "pretrained_models/MobileNetV2_mAP=0.383.pth"


@dataclass
class PLResNet38Config(AudioModelConfig):
    """Configuration for ResNet38 models.

    Args:
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        metrics_threshold: Threshold for metrics calculation
        lr_scheduler_patience: Patience for learning rate scheduler
        sample_rate: Sample rate of audio
        window_size: Window size for spectrogram calculation
        hop_size: Hop size for spectrogram calculation
        mel_bins: Number of mel bins
        fmin: Minimum frequency
        fmax: Maximum frequency
        classes_num: Number of classes
        checkpoint_path: Path to checkpoint
    """

    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    metrics_threshold: float = 0.5
    checkpoint_path: str | None = "pretrained_models/ResNet38_mAP=0.434.pth"
