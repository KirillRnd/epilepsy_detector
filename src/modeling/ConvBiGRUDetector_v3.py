import torch
import torch.nn as nn
from .model_registry import register_model


@register_model("ConvBiGRUDetector_v3")
class ConvBiGRUDetector_v3(nn.Module):
    def __init__(self, input_channels=3, conv_hidden=16,
                 gru_hidden=32, gru_layers=1, dropout=0.15):
        super().__init__()

        # Нормализация входа: BatchNorm по (batch, T)
        # running_mean/var калибруются на train, применяются на test
        # Сохраняет относительные различия между окнами
        self.input_norm = nn.BatchNorm1d(input_channels)

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, conv_hidden,
                      kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(conv_hidden, conv_hidden,
                      kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=conv_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x):
        x = self.input_norm(x)       # нормализует, но сохраняет различия
        x = self.features(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.head(x).squeeze(-1)
        return x
