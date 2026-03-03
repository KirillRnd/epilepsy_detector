import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_registry import register_model

class TCNBlock(nn.Module):
    """Один блок TCN: dilated causal conv + residual."""
    def __init__(self, channels, kernel_size=5, dilation=1, dropout=0.15):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2  # same padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.drop(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.bn2(self.conv2(x))))
        return x + residual

@register_model("TCNDetector")
class TCNDetector(nn.Module):
    """
    Dilated TCN для покадровой детекции приступов.
    Вход:  (B, 3, 2000)
    Выход: (B, 2000) logits
    ~10-20K параметров при hidden=16, 4 блока
    """
    def __init__(self, input_channels=3, hidden=16, n_blocks=4,
                 kernel_size=5, dropout=0.15):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.ModuleList([
            TCNBlock(hidden, kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_blocks)
        ])
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)

    def forward(self, x):
        x = self.input_proj(x)        # (B, hidden, T)
        for block in self.blocks:
            x = block(x)              # (B, hidden, T)
        x = self.head(x).squeeze(1)   # (B, T)
        return x
