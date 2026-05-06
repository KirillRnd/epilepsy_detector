import torch
import torch.nn as nn
import torch.nn.functional as F
from .model_registry import register_model


class RobustChannelNorm(nn.Module):
    """
    Робастная нормализация каждого канала по усечённому среднему и IQR.

    Вместо median/MAD используем:
      - center: (q25 + q75) / 2  — midquartile, детерминирован на CUDA
      - scale:  IQR * 0.7413     — согласованная оценка std

    torch.quantile детерминирован на CUDA в отличие от torch.median.

    Вход/выход: (B, C, T)
    """

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_channels, 1))
        self.bias   = nn.Parameter(torch.zeros(1, num_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        q25 = torch.quantile(x, 0.25, dim=2, keepdim=True)  # (B, C, 1)
        q75 = torch.quantile(x, 0.75, dim=2, keepdim=True)  # (B, C, 1)

        center = (q25 + q75) / 2.0        # midquartile ≈ медиана
        scale  = (q75 - q25) * 0.7413     # IQR → согласованная оценка std

        x_norm = (x - center) / (scale + self.eps)

        return x_norm * self.weight + self.bias


# ---------------------------------------------------------------------------

class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation: адаптивная рекалибровка каналов."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.squeeze   = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1)
        return x * s


class RDSCBlock(nn.Module):
    """Residual Depthwise Separable Conv + SE attention."""

    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.15):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv1d(channels, channels, kernel_size,
                      padding=pad, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.se = SEBlock1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        return out + x  # residual


# ---------------------------------------------------------------------------

@register_model("RDSCBiGRUDetector_v2")
class RDSCBiGRUDetector_v2(nn.Module):
    """
    RDSCBiGRUDetector с встроенной робастной нормализацией (вариант C).

    Отличие от RDSCBiGRUDetector:
      — Первый слой: RobustChannelNorm (медиана + MAD по временной оси)
        вместо отсутствия нормализации.
      — Код препроцессинга не изменён, можно сравнивать модели напрямую.

    Вход:  (B, 3, 2000)
    Выход: (B, 2000) logits
    """

    def __init__(
        self,
        input_channels: int = 3,
        hidden: int = 32,
        gru_hidden: int = 32,
        gru_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()

        # 0. Робастная нормализация входа
        #    stop_gradient=False — нормализация дифференцируема,
        #    градиент проходит через learnable weight/bias
        self.input_norm = RobustChannelNorm(input_channels)

        # 1. Stem: расширяем каналы обычной свёрткой
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        # 2. RDSC блоки с SE-attention (идентично оригиналу)
        self.rdsc_blocks = nn.Sequential(
            RDSCBlock(hidden, kernel_size=9, dropout=dropout),
            RDSCBlock(hidden, kernel_size=9, dropout=dropout),
            RDSCBlock(hidden, kernel_size=7, dropout=dropout),
        )

        # 3. Temporal smoother
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # 4. Per-frame classifier
        self.head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.input_norm(x)      # (B, C, T) — робастная нормализация
        x = self.stem(x)            # (B, hidden, T)
        x = self.rdsc_blocks(x)     # (B, hidden, T)
        x = x.permute(0, 2, 1)     # (B, T, hidden)
        x, _ = self.gru(x)         # (B, T, gru_hidden*2)
        x = self.head(x).squeeze(-1)  # (B, T)
        return x
