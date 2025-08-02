"""Factory for creating model configurations."""

from typing import Any

from src.config.model_config import (
    AudioModelConfig,
    PLASTConfig,
    PLCnn14Config,
    PLDaiNet19Config,
    PLLeeNet24Config,
    PLMobileNetV1Config,
    PLMobileNetV2Config,
    PLResNet38Config,
)


def create_model_config(model_type: str, **kwargs: dict[str, Any]) -> AudioModelConfig:
    """Create a model configuration based on model type.

    Args:
        model_type: Type of model architecture
        **kwargs: Additional keyword arguments to pass to the model configuration

    Returns:
        Model configuration instance
    """
    model_config_classes: dict[str, type[AudioModelConfig]] = {
        "ast": PLASTConfig,
        "cnn14": PLCnn14Config,
        "dainet19": PLDaiNet19Config,
        "leenet24": PLLeeNet24Config,
        "mobilenetv1": PLMobileNetV1Config,
        "mobilenetv2": PLMobileNetV2Config,
        "resnet38": PLResNet38Config,
    }

    if model_type not in model_config_classes:
        raise ValueError(f"Unsupported model type: {model_type}")

    config_class = model_config_classes[model_type]
    return config_class(**kwargs)
