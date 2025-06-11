import torch

from src.architectures.denoise.BaseDenoiserLightningModule import BaseDenoiserLightningModule

class AdaptiveKalman(BaseDenoiserLightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Filtro de Kalman adaptativo para redução de ruído."""

        x_est = torch.zeros_like(x)
        error_est = torch.zeros_like(x)

        x_est[0] = x[0]
        error_est[0] = 1.0
        current_measurement_noise = self.initial_measurement_noise

        for k in range(1, len(x)):
            if k % self.adaptation_interval == 0:
                current_measurement_noise = torch.var(
                    x[max(0, k - self.adaptation_interval) : k],
                    unbiased=False
                )

            x_pred = x_est[k - 1]
            error_pred = error_est[k - 1] + self.process_variance

            K = error_pred / (error_pred + current_measurement_noise)
            x_est[k] = x_pred + K * (x[k] - x_pred)
            error_est[k] = (1 - K) * error_pred

        return x_est
