import numpy as np
import torch

from src.architectures.denoise.BaseDenoiserLightningModule import (
    BaseDenoiserLightningModule,
)


class AdaptiveKalman(BaseDenoiserLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Filtro de Kalman adaptativo para redução de ruído."""
        y = x.cpu().numpy() if x.is_cuda else x.numpy()
        process_variance = self.process_variance
        initial_measurement_noise = self.initial_measurement_noise
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

        x_denoised = torch.from_numpy(x_est)
        assert x_denoised.shape == x.shape, "Output shape mismatch"

        return x_denoised.to(x.device)
