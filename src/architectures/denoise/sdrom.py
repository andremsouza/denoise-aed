import os
from typing import Optional, Tuple

import librosa
import mlflow
import numpy as np
import soundfile as sf

from src.utils.metrics import calculate_snr, calculate_stoi, calculate_sdr


class SDROM:
    def __init__(self, sr: int = 16000, frame_size: int = 1024, hop_size: int = 512):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size

    def process(
        self,
        y: np.ndarray,
        alpha: float = 0.95,
        beta: float = 0.98,
        adaptive: bool = True,
    ) -> np.ndarray:
        """Aplica filtro SDROM com parâmetros adaptativos."""
        stft = librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_size)
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

        return librosa.istft(magnitude * np.exp(1j * phase), hop_length=self.hop_size)

    @staticmethod
    def train(
        input_path: str,
        output_dir: str,
        alpha: float = 0.95,
        beta: float = 0.98,
        mode: str = "snr",
    ) -> Tuple[str, dict]:
        """Executa o processamento e avaliação."""
        with mlflow.start_run():
            processor = SDROM()
            y, sr = librosa.load(input_path, sr=None)
            y_clean = processor.process(y, alpha, beta)

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"sdrom_{mode}_{os.path.basename(input_path)}"
            )
            sf.write(output_path, y_clean, sr)

            metrics = {
                "snr": calculate_snr(y, y_clean),
                "stoi": calculate_stoi(y, y_clean, sr),
                "sdr": calculate_sdr(y, y_clean),
            }

            mlflow.log_params({"alpha": alpha, "beta": beta, "adaptive": True})
            mlflow.log_metrics(metrics)

            return output_path, metrics
