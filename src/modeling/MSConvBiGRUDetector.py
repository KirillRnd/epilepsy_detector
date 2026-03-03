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

@register_model("MSConvBiGRUDetector")
class MSConvBiGRUDetector(nn.Module):
    """
    Multi-Scale Conv + Channel Attention + BiGRU.
    Три параллельных ветки свёрток с разными kernel_size
    захватывают паттерны на разных временных масштабах.
    """
    def __init__(self, input_channels=3, branch_hidden=8,
                 gru_hidden=32, gru_layers=1, dropout=0.15):
        super().__init__()
        # Три ветки с разными рецептивными полями
        self.branch_short = nn.Sequential(
            nn.Conv1d(input_channels, branch_hidden,
                      kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(branch_hidden),
            nn.ReLU(inplace=True),
        )
        self.branch_mid = nn.Sequential(
            nn.Conv1d(input_channels, branch_hidden,
                      kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(branch_hidden),
            nn.ReLU(inplace=True),
        )
        self.branch_long = nn.Sequential(
            nn.Conv1d(input_channels, branch_hidden,
                      kernel_size=31, padding=15, bias=False),
            nn.BatchNorm1d(branch_hidden),
            nn.ReLU(inplace=True),
        )

        total_channels = branch_hidden * 3  # 24

        # Channel attention: SE блок после конкатенации
        self.se = SEBlock1d(total_channels, reduction=4)

        # Дополнительная свёртка для смешивания
        self.mixer = nn.Sequential(
            nn.Conv1d(total_channels, total_channels,
                      kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(total_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # BiGRU
        self.gru = nn.GRU(
            input_size=total_channels,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x):
        # x: (B, C, T)
        s = self.branch_short(x)  # (B, 8, T)
        m = self.branch_mid(x)    # (B, 8, T)
        l = self.branch_long(x)   # (B, 8, T)

        x = torch.cat([s, m, l], dim=1)  # (B, 24, T)
        x = self.se(x)                    # channel attention
        x = self.mixer(x)                 # (B, 24, T)

        x = x.permute(0, 2, 1)    # (B, T, 24)
        x, _ = self.gru(x)        # (B, T, 64)
        x = self.head(x).squeeze(-1)  # (B, T)
        return x
