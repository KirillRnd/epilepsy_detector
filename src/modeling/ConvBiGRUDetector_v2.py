import torch
import torch.nn as nn
from .model_registry import register_model


@register_model("ConvBiGRUDetector_v2")
class ConvBiGRUDetector_v2(nn.Module):
    """
    Conv1d feature extractor + BiGRU temporal smoother
    со встроенной InstanceNorm для cross-subject нормализации.

    Вход:  (B, 3, 2000) — сырой ЭЭГ, любой масштаб амплитуды
    Выход: (B, 2000) logits
    """

    def __init__(self, input_channels=3, conv_hidden=16,
                 gru_hidden=32, gru_layers=1, dropout=0.15):
        super().__init__()

        # ---- Встроенная нормализация ----
        # InstanceNorm1d: для каждого (sample, channel) нормализует по T
        # affine=True — обучаемые gamma/beta per channel
        self.input_norm = nn.InstanceNorm1d(
            input_channels, affine=True
        )

        # ---- Feature extractor ----
        # GroupNorm вместо BatchNorm: нормализует по группам каналов
        # внутри каждого семпла, не зависит от batch size и running stats
        num_groups_1 = min(4, conv_hidden)  # 4 группы для 16 каналов
        num_groups_2 = min(4, conv_hidden)
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, conv_hidden,
                      kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(num_groups_1, conv_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(conv_hidden, conv_hidden,
                      kernel_size=9, padding=4, bias=False),
            nn.GroupNorm(num_groups_2, conv_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # ---- Temporal smoother ----
        self.gru = nn.GRU(
            input_size=conv_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # ---- Per-frame classifier ----
        self.head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) — сырой сигнал любого масштаба
        x = self.input_norm(x)       # (B, C, T) — нормализовано per-sample
        x = self.features(x)         # (B, conv_hidden, T)
        x = x.permute(0, 2, 1)       # (B, T, conv_hidden)
        x, _ = self.gru(x)           # (B, T, gru_hidden*2)
        x = self.head(x).squeeze(-1) # (B, T)
        return x
