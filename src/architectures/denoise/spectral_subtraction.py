import librosa
import numpy as np
import torch

from src.architectures.denoise.BaseDenoiserLightningModule import (
    BaseDenoiserLightningModule,
)


class SpectralSubtraction(BaseDenoiserLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Aplica subtração espectral com parâmetros ajustáveis."""
        y = x.cpu().numpy() if x.is_cuda else x.numpy()
        noise_reduction_factor = self.noise_reduction_factor
        noise_window_duration = self.noise_window_duration

        stft = librosa.stft(y, n_fft=self.window_size, hop_length=self.hop_size)
        magnitude, phase = np.abs(stft), np.angle(stft)

        noise_frames = int(noise_window_duration * self.sample_rate / self.hop_size)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        subtracted = np.maximum(magnitude - noise_reduction_factor * noise_spectrum, 0)

        x_denoised = torch.from_numpy(
            librosa.istft(subtracted * np.exp(1j * phase), hop_length=self.hop_size)
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
