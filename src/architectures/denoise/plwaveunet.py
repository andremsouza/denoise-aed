import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.audio import SignalNoiseRatio, ScaleInvariantSignalNoiseRatio
from torchmetrics.audio.stoi import ShortTimeObjectiveIntelligibility


class ConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, is_transpose=False
    ):
        super().__init__()
        if is_transpose:
            self.conv = nn.ConvTranspose1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                output_padding=stride - 1 if stride > 1 else 0,
            )
        else:
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
            )
        self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return F.leaky_relu(self.norm(self.conv(x)), negative_slope=0.1)


class WaveUNet(pl.LightningModule):
    def __init__(self, num_channels=12, kernel_size=5, stride=2, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.encoder_channels = [2**i for i in range(num_channels)]
        self.encoders = nn.ModuleList()

        # Build encoders
        in_ch = 1
        for out_ch in self.encoder_channels:
            self.encoders.append(ConvBlock(in_ch, out_ch, kernel_size, stride))
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(
            self.encoder_channels[-1],
            self.encoder_channels[-1] * 2,
            kernel_size,
            stride,
        )

        # Build decoders (tracking channels correctly)
        self.decoders = nn.ModuleList()
        decoder_in_chs = []
        decoder_out_chs = list(reversed(self.encoder_channels[:-1]))

        # Define input channels to decoder blocks based on skip + x
        x_ch = self.encoder_channels[-1] * 2
        for skip_ch, out_ch in zip(
            reversed(self.encoder_channels[:-1]), decoder_out_chs
        ):
            self.decoders.append(
                ConvBlock(
                    x_ch + skip_ch, out_ch, kernel_size, stride, is_transpose=True
                )
            )
            x_ch = out_ch  # Update x for next iteration

        # Final conv layer
        self.final_conv = nn.Conv1d(x_ch, 1, kernel_size=1)

        self.metrics = nn.ModuleDict(
            {
                "snr": SignalNoiseRatio(),
                "si_snr": ScaleInvariantSignalNoiseRatio(),
                "stoi": ShortTimeObjectiveIntelligibility(16000, extended=False),
            }
        )

        self.loss_fn = nn.L1Loss()

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 4 and x.shape[2] == 1:
            x = x.squeeze(2)

        skips: list[torch.Tensor] = []

        # Encoder
        for enc in self.encoders:
            x = enc(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i, dec in enumerate(self.decoders):
            skip = skips[-i - 2]
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(
                    x, size=skip.shape[-1], mode="linear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        x = self.final_conv(x).squeeze(1)
        return x

    def _shared_step(self, batch, prefix):
        noisy, clean = batch

        if noisy.dim() == 4 and noisy.shape[2] == 1:
            noisy = noisy.squeeze(2)
        if clean.dim() == 4 and clean.shape[2] == 1:
            clean = clean.squeeze(2)

        pred = self(noisy)

        # Garantir shapes compatíveis
        if pred.dim() == 2 and clean.dim() == 3:
            clean = clean.squeeze(1)
        elif pred.dim() == 3 and clean.dim() == 2:
            pred = pred.squeeze(1)

        if pred.shape != clean.shape:
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=clean.shape[-1],
                mode="linear",
                align_corners=False,
            ).squeeze(1)

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
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
