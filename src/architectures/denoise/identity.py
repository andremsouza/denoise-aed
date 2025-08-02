import torch

from src.architectures.denoise.BaseDenoiserLightningModule import BaseDenoiserLightningModule

class IdentityDenoiser(BaseDenoiserLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x

