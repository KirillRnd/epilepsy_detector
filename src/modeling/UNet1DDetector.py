import torch
import torch.nn as nn

from .model_registry import register_model  
class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.15):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x  # pooled, skip

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.15):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_ch * 2, out_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x, skip):
        x = self.up(x)
        # Обрезка skip, если длины не совпадают
        if x.size(2) != skip.size(2):
            skip = skip[:, :, :x.size(2)]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

@register_model("UNet1DDetector")
class UNet1DDetector(nn.Module):
    """
    Encoder-Decoder с skip connections для 1D сегментации.
    Вход:  (B, 3, 2000)
    Выход: (B, 2000) logits
    ~20-40K параметров при базовом ch=16, 3 уровня
    """
    def __init__(self, input_channels=3, base_ch=16, n_levels=3, dropout=0.15):
        super().__init__()
        channels = [base_ch * (2 ** i) for i in range(n_levels + 1)]
        # channels: [16, 32, 64, 128] при base_ch=16, n_levels=3

        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, channels[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.encoders = nn.ModuleList([
            DownBlock(channels[i], channels[i+1], dropout) for i in range(n_levels)
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv1d(channels[-1], channels[-1], kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(channels[-1]),
            nn.ReLU(inplace=True),
        )
        self.decoders = nn.ModuleList([
            UpBlock(channels[i+1], channels[i], dropout) for i in reversed(range(n_levels))
        ])
        self.head = nn.Conv1d(channels[0], 1, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)
        skips = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)
        x = self.head(x).squeeze(1)
        # Обрезка до исходной длины
        return x[:, :2000] if x.size(1) > 2000 else x
