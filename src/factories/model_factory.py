"""Model factory for creating Lightning modules."""

import lightning.pytorch as pl

from src.architectures.plast import PLAST
from src.architectures.plcnn14 import PLCnn14
from src.architectures.pldainet import PLDaiNet19
from src.architectures.plleenet import PLLeeNet24
from src.architectures.plmobilenetv1 import PLMobileNetV1
from src.architectures.plmobilenetv2 import PLMobileNetV2
from src.architectures.plresnet import PLResNet38
from src.config.model_config import AudioModelConfig

# Registry of available models
MODEL_REGISTRY: dict[str, type] = {
    "ast": PLAST,
    "cnn14": PLCnn14,
    "dainet19": PLDaiNet19,
    "leenet24": PLLeeNet24,
    "mobilenetv1": PLMobileNetV1,
    "mobilenetv2": PLMobileNetV2,
    "resnet38": PLResNet38,
}


def create_model(model_type: str, config: AudioModelConfig) -> pl.LightningModule:
    """Create a model instance.

    Args:
        model_type: Type of the model to create
        config: Model configuration

    Returns:
        Instantiated model

    Raises:
        ValueError: If model_type is not in the registry
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model: {model_type}. Available models: {list(MODEL_REGISTRY.keys())}"
        )

    model_class = MODEL_REGISTRY[model_type]
    return model_class(**vars(config))
