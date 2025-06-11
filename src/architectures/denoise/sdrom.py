import torch
import math as m
from torchlibrosa.stft import STFT, ISTFT

from src.architectures.denoise.BaseDenoiserLightningModule import BaseDenoiserLightningModule


class SDROM(BaseDenoiserLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stft_fn = STFT(
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size)

        self.istft_fn = ISTFT(
            n_fft=self.window_size,
            hop_length=self.hop_size,
            win_length=self.window_size
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Aplica filtro SDROM com parâmetros adaptativos."""

        stft = self.stft_fn(x)

        magnitude, phase = torch.abs(stft), torch.angle(stft)
        mean_magnitude = torch.zeros_like(magnitude[:, 0])

        for i in range(magnitude.shape[1]):
            curr_alpha = self.alpha / (1 + m.log1p(i)) if self.adaptive else self.alpha
            curr_beta = self.beta / (1 + m.log1p(i)) if self.adaptive else self.beta

            mean_magnitude = (
                curr_alpha * mean_magnitude + (1 - curr_alpha) * magnitude[:, i]
            )
            magnitude[:, i] = torch.maximum(
                magnitude[:, i] - curr_beta * mean_magnitude,
                torch.zeros_like(magnitude[:, i])
            )

        reconstructed = magnitude * torch.exp(1j * phase)
        return self.istft_fn(reconstructed)