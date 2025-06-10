import os
from typing import Tuple

import librosa
import mlflow
import numpy as np
import soundfile as sf

from src.utils.metrics import calculate_snr, calculate_stoi, calculate_sdr
from src.utils.audio_utils import normalize_audio


class AdaptiveKalman:
    def __init__(self, adaptation_interval: int = 100):
        self.adaptation_interval = adaptation_interval

    def process(
        self,
        y: np.ndarray,
        process_variance: float = 1e-5,
        initial_measurement_noise: float = 1e-4,
    ) -> np.ndarray:
        """Filtro de Kalman adaptativo para redução de ruído."""
        x_est = np.zeros_like(y)
        error_est = np.zeros_like(y)

        x_est[0] = y[0]
        error_est[0] = 1.0
        current_measurement_noise = initial_measurement_noise

        for k in range(1, len(y)):
            if k % self.adaptation_interval == 0:
                current_measurement_noise = np.var(
                    y[max(0, k - self.adaptation_interval) : k]
                )

            x_pred = x_est[k - 1]
            error_pred = error_est[k - 1] + process_variance

            K = error_pred / (error_pred + current_measurement_noise)
            x_est[k] = x_pred + K * (y[k] - x_pred)
            error_est[k] = (1 - K) * error_pred

        return x_est

    @staticmethod
    def train(
        input_path: str,
        output_dir: str,
        process_variance: float = 1e-5,
        measurement_noise: float = 1e-4,
        mode: str = "snr",
    ) -> Tuple[str, dict]:
        """Processa e avalia o filtro de Kalman."""
        with mlflow.start_run():
            y, sr = librosa.load(input_path, sr=None)
            y_norm = normalize_audio(y)

            processor = AdaptiveKalman()
            y_clean = processor.process(y_norm, process_variance, measurement_noise)
            y_clean = y_clean * np.max(np.abs(y))  # Restaura escala original

            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(
                output_dir, f"kalman_{mode}_{os.path.basename(input_path)}"
            )
            sf.write(output_path, y_clean, sr)

            metrics = {
                "snr": calculate_snr(y, y_clean),
                "stoi": calculate_stoi(y, y_clean, sr),
                "sdr": calculate_sdr(y, y_clean),
            }

            mlflow.log_params(
                {
                    "process_variance": process_variance,
                    "measurement_noise": measurement_noise,
                }
            )
            mlflow.log_metrics(metrics)

            return output_path, metrics
