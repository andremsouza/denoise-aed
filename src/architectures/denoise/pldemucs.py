import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class Demucs(pl.LightningModule):
    def __init__(
        self,
        encoder_channels: list[int] = [64, 128, 256, 512],
        kernel_size: int = 8,
        stride: int = 4,
        lr: float = 1e-4,
        input_channels: int = 1,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Camadas de adaptação de canais
        self.input_conv = nn.Conv1d(input_channels, encoder_channels[0], kernel_size=1)

        # Encoder/Decoder
        self.encoder = self._build_blocks(
            encoder_channels, kernel_size, stride, is_encoder=True
        )
        self.decoder = self._build_blocks(
            encoder_channels[::-1], kernel_size, stride, is_encoder=False
        )

        self.output_conv = nn.Conv1d(encoder_channels[0], input_channels, kernel_size=1)

        # Metrics
        self.metrics = nn.ModuleDict(
            {
                "snr": SignalNoiseRatio(),
                "si_snr": ScaleInvariantSignalNoiseRatio(),
                "stoi": ShortTimeObjectiveIntelligibility(16000, extended=False),
            }
        )
        self.loss_fn = nn.L1Loss()

    def _build_blocks(
        self, channels: list[int], kernel_size: int, stride: int, is_encoder: bool
    ) -> nn.Sequential:
        blocks = []
        for i in range(len(channels) - 1):
            padding = (kernel_size - stride) // 2  # Padding para manter dimensões
            if is_encoder:
                conv1d = nn.Conv1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
                blocks.extend([conv1d, nn.ReLU()])
            else:
                convtranspose1d = nn.ConvTranspose1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    output_padding=stride // 2,  # Ajuste fino para dimensões
                )
                blocks.extend([convtranspose1d, nn.ReLU()])
        return nn.Sequential(*blocks)

    def adjust_length(self, x, target_length):
        """Ajusta o comprimento do tensor para match com o target"""
        if x.shape[-1] < target_length:
            # Padding se menor
            return F.pad(x, (0, target_length - x.shape[-1]))
        elif x.shape[-1] > target_length:
            # Cortar se maior
            return x[..., :target_length]
        return x

    def forward(self, x, target_length=None):
        # Guarda o comprimento original se fornecido
        orig_length = x.shape[-1] if target_length is None else target_length

        # Garantir shape [B, C, T]
        if x.dim() == 4:
            x = x.squeeze(1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.output_conv(x)

        # Ajuste final do comprimento
        return self.adjust_length(x, orig_length)

    def _shared_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], prefix: str
    ) -> torch.Tensor:
        noisy, clean = batch

        # Ajuste para garantir que clean tenha o mesmo shape que pred
        if clean.dim() == 4 and clean.shape[2] == 1:
            clean = clean.squeeze(2)

        pred = self(noisy, target_length=clean.shape[-1])
        loss = self.loss_fn(pred, clean)

        metrics = {f"{prefix}_loss": loss}
        for name, metric in self.metrics.items():
            metrics[f"{prefix}_{name}"] = metric(pred, clean)

        self.log_dict(metrics, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        # Chamar _shared_step (retornando apenas loss)
        loss = self._shared_step(batch, "val")

        # Registrar a perda de validação
        self.log("val_loss", loss, prog_bar=True, logger=True)

        return {"val_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
