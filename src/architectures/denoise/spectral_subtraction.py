from torchlibrosa.stft import STFT, ISTFT
import torch

from src.architectures.denoise.BaseDenoiserLightningModule import BaseDenoiserLightningModule

class SpectralSubtraction(BaseDenoiserLightningModule):
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
        """Aplica subtração espectral com parâmetros ajustáveis."""
        stft = self.stft_fn(x)
        magnitude, phase = torch.abs(stft), torch.angle(stft)

        noise_frames = int(self.noise_window_duration * self.sample_rate/ self.hop_size)
        noise_spectrum = torch.mean(magnitude[:, :noise_frames], dim=1, keepdim=True)

        subtracted = torch.maximum(magnitude - self.noise_reduction_factor * noise_spectrum, torch.zeros_like(magnitude))

        reconstructed = subtracted * torch.exp(1j * phase)
        return self.istft_fn(reconstructed)
