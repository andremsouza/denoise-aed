import torch
import librosa
import numpy as np

from src.architectures.denoise.BaseDenoiserLightningModule import (
    BaseDenoiserLightningModule,
)


class SDROM(BaseDenoiserLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Aplica filtro SDROM com parâmetros adaptativos."""
        alpha = self.alpha
        beta = self.beta
        adaptive = self.adaptive
        y = x.cpu().numpy() if x.is_cuda else x.numpy()
        stft = librosa.stft(y, n_fft=self.window_size, hop_length=self.hop_size)
        magnitude, phase = np.abs(stft), np.angle(stft)
        mean_magnitude = np.zeros_like(magnitude[:, 0])

        for i in range(magnitude.shape[1]):
            curr_alpha = alpha / (1 + np.log1p(i)) if adaptive else alpha
            curr_beta = beta / (1 + np.log1p(i)) if adaptive else beta

            mean_magnitude = (
                curr_alpha * mean_magnitude + (1 - curr_alpha) * magnitude[:, i]
            )
            magnitude[:, i] = np.maximum(
                magnitude[:, i] - curr_beta * mean_magnitude, 0
            )

        x_denoised = torch.from_numpy(
            librosa.istft(magnitude * np.exp(1j * phase), hop_length=self.hop_size)
        )
        # Ensure the output shape matches the input shape
        # If x_denoised is larger than x, we trim it
        if x_denoised.shape[0] > x.shape[0]:
            x_denoised = x_denoised[: x.shape[0]]
        elif x_denoised.shape[0] < x.shape[0]:
            # If x_denoised is smaller, we pad it
            padding = x.shape[0] - x_denoised.shape[0]
            x_denoised = torch.nn.functional.pad(x_denoised, (0, padding))
        # Check if the output shape matches the input shape
        assert x_denoised.shape == x.shape, "Output shape mismatch"

        return x_denoised.to(x.device)
