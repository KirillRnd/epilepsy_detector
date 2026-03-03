import torch
import torch.nn as nn
from .model_registry import register_model


class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation: адаптивная рекалибровка каналов."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.shape
        s = self.squeeze(x).view(b, c)          # (B, C)
        s = self.excitation(s).view(b, c, 1)     # (B, C, 1)
        return x * s


class RDSCBlock(nn.Module):
    """Residual Depthwise Separable Conv + SE attention."""
    def __init__(self, channels, kernel_size=9, dropout=0.15):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            # Depthwise: отдельная свёртка для каждого канала
            nn.Conv1d(channels, channels, kernel_size,
                      padding=pad, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # Pointwise: 1x1 свёртка для смешивания каналов
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.se = SEBlock1d(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out + x  # Residual connection


@register_model("RDSCBiGRUDetector")
class RDSCBiGRUDetector(nn.Module):
    """
    Residual Depthwise Separable Conv + SE Attention + BiGRU.
    Архитектура по мотивам RDSC-GRU (2026) для cross-subject seizure detection.

    Вход:  (B, 3, 2000)
    Выход: (B, 2000) logits
    """
    def __init__(self, input_channels=3, hidden=32,
                 gru_hidden=32, gru_layers=1, dropout=0.15):
        super().__init__()

        # Stem: расширяем каналы обычной свёрткой
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden,
                      kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        # RDSC блоки с SE-attention
        self.rdsc_blocks = nn.Sequential(
            RDSCBlock(hidden, kernel_size=9, dropout=dropout),
            RDSCBlock(hidden, kernel_size=9, dropout=dropout),
            RDSCBlock(hidden, kernel_size=7, dropout=dropout),
        )

        # Temporal smoother
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # Per-frame classifier
        self.head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x):
        x = self.stem(x)              # (B, hidden, T)
        x = self.rdsc_blocks(x)       # (B, hidden, T) — с residuals
        x = x.permute(0, 2, 1)        # (B, T, hidden)
        x, _ = self.gru(x)            # (B, T, gru_hidden*2)
        x = self.head(x).squeeze(-1)  # (B, T)
        return x
