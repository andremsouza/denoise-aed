import os
from typing import Optional, Tuple

import librosa
import mlflow
import numpy as np
import soundfile as sf

from src.utils.metrics import calculate_snr, calculate_stoi, calculate_sdr
from src.utils.audio_utils import normalize_audio


class SpectralSubtraction:
    def __init__(self, sr: int = 16000, frame_size: int = 1024, hop_size: int = 512):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size

    def process(
        self,
        y: np.ndarray,
        noise_reduction_factor: float = 1.0,
        noise_window_duration: float = 0.1,
    ) -> np.ndarray:
        """Aplica subtração espectral com parâmetros ajustáveis."""
        stft = librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_size)
        magnitude, phase = np.abs(stft), np.angle(stft)

        noise_frames = int(noise_window_duration * self.sr / self.hop_size)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)

        subtracted = np.maximum(magnitude - noise_reduction_factor * noise_spectrum, 0)
        return librosa.istft(subtracted * np.exp(1j * phase), hop_length=self.hop_size)

    @staticmethod
    def train(
        input_path: str,
        output_dir: str,
        noise_reduction_factor: float = 1.0,
        noise_window_duration: float = 0.1,
        mode: str = "snr",
    ) -> Tuple[str, dict]:
        """Processa um arquivo e retorna métricas."""
        with mlflow.start_run():
            processor = SpectralSubtraction()
            y, sr = librosa.load(input_path, sr=None)
            y_clean = processor.process(
                y, noise_reduction_factor, noise_window_duration
            )

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"spectral_{mode}_{os.path.basename(input_path)}"
            )
            sf.write(output_path, y_clean, sr)

            metrics = {
                "snr": calculate_snr(y, y_clean),
                "stoi": calculate_stoi(y, y_clean, sr),
                "sdr": calculate_sdr(y, y_clean),
            }

            mlflow.log_params(
                {
                    "noise_reduction_factor": noise_reduction_factor,
                    "noise_window_duration": noise_window_duration,
                }
            )
            mlflow.log_metrics(metrics)

            return output_path, metrics
