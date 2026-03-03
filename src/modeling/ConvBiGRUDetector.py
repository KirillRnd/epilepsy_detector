import torch
import torch.nn as nn
from .model_registry import register_model

@register_model("ConvBiGRUDetector")
class ConvBiGRUDetector(nn.Module):
    """
    Conv1d feature extractor + BiGRU temporal smoother.
    Вход:  (B, 3, 2000)
    Выход: (B, 2000) logits
    ~15-25K параметров при hidden=16, gru_hidden=32
    """
    def __init__(self, input_channels=3, conv_hidden=16,
                 gru_hidden=32, gru_layers=1, dropout=0.15):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, conv_hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(conv_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(conv_hidden, conv_hidden, kernel_size=9, padding=4, bias=False),
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
        self.head = nn.Linear(gru_hidden * 2, 1)  # *2 от bidirectional

    def forward(self, x):
        x = self.features(x)         # (B, conv_hidden, T)
        x = x.permute(0, 2, 1)       # (B, T, conv_hidden)
        x, _ = self.gru(x)           # (B, T, gru_hidden*2)
        x = self.head(x).squeeze(-1) # (B, T)
        return x
