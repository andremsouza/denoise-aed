import lightning.pytorch as pl
import torch
import torch.nn as nn
from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class ConvTasNet(pl.LightningModule):
    def __init__(
        self,
        kernel_size: int = 16,
        num_filters: int = 512,
        separator_layers: int = 4,
        lr: float = 3e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Encoder/Decoder
        self.encoder = nn.Conv1d(
            1, num_filters, kernel_size=kernel_size, stride=kernel_size // 2, padding=0
        )

        # Separator
        self.separator = nn.Sequential(
            *[
                nn.Sequential(
                    nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1),
                    nn.BatchNorm1d(num_filters),
                    nn.PReLU(),
                    nn.Dropout(0.1),
                )
                for _ in range(separator_layers)
            ]
        )

        self.decoder = nn.ConvTranspose1d(
            num_filters, 1, kernel_size=kernel_size, stride=kernel_size // 2, padding=0
        )

        # Metrics
        self.metrics = nn.ModuleDict(
            {
                "snr": SignalNoiseRatio(),
                "si_snr": ScaleInvariantSignalNoiseRatio(),
                "stoi": ShortTimeObjectiveIntelligibility(16000, extended=False),
            }
        )
        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        """Propagação direta."""

        # Garantir que a entrada esteja no formato correto para Conv1D
        x = (
            x.squeeze(2) if x.dim() == 4 else x
        )  # Remove dimensões extras (ajustando de [8, 1, 1, 32000] para [8, 1, 32000])

        # Propagação usual
        encoded = self.encoder(x)
        separated = self.separator(encoded)
        decoded = self.decoder(separated)
        return decoded

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # Chamar _shared_step (retornando apenas loss)
        loss = self._shared_step(batch, "val")

        # Registrar a perda de validação
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {"val_loss": loss}

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        noisy, clean = batch
        pred = self(noisy)

        # Garantir shapes compatíveis
        if pred.shape != clean.shape:
            pred = torch.nn.functional.interpolate(
                pred, size=clean.shape[-1], mode="linear", align_corners=False
            )

        loss = self.loss_fn(pred, clean)

        metrics = {f"{prefix}_loss": loss}
        for name, metric in self.metrics.items():
            metrics[f"{prefix}_{name}"] = metric(pred, clean)

        self.log_dict(metrics, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-5
        )
