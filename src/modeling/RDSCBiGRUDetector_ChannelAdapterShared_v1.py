import torch
import torch.nn as nn
from .model_registry import register_model


class SEBlock1d(nn.Module):
    """Squeeze-and-Excitation: адаптивная рекалибровка hidden-каналов."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, _ = x.shape
        s = self.squeeze(x).view(b, c)       # (B, C)
        s = self.excitation(s).view(b, c, 1) # (B, C, 1)
        return x * s


class RDSCBlock(nn.Module):
    """Residual Depthwise Separable Conv + SE attention."""
    def __init__(self, channels, kernel_size=9, dropout=0.15):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            # Depthwise: отдельная свёртка для каждого hidden feature map.
            nn.Conv1d(channels, channels, kernel_size,
                      padding=pad, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            # Pointwise: 1x1 свёртка для смешивания hidden feature maps.
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.se = SEBlock1d(channels)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out + x


class SharedChannelAdapter1d(nn.Module):
    """
    Маленькая channel-aware residual-ветка со shared weights.

    Идея:
      - один и тот же Conv1d(1 -> channel_hidden) применяется к каждому
        физическому каналу EEG отдельно;
      - затем добавляется минимальная per-channel affine-поправка
        (gain/bias), чтобы FrL/FrR/OcR_Hipp могли отличаться;
      - после concat выполняется 1x1 mix в hidden-размер основного stem.

    Ветка не нормализует raw input и не заменяет основной stem.
    """
    def __init__(
        self,
        input_channels=3,
        channel_hidden=4,
        hidden=32,
        kernel_size=7,
        dropout=0.05,
    ):
        super().__init__()
        self.input_channels = input_channels
        self.channel_hidden = channel_hidden
        pad = kernel_size // 2

        # Shared фильтробанк: одни и те же waveform-фильтры для всех
        # физических каналов.
        self.shared_conv = nn.Conv1d(
            1,
            channel_hidden,
            kernel_size=kernel_size,
            padding=pad,
            bias=True,
        )
        self.act = nn.ReLU(inplace=True)

        # Очень маленькая channel-specific часть.
        self.channel_gain = nn.Parameter(
            torch.ones(1, input_channels, channel_hidden, 1)
        )
        self.channel_bias = nn.Parameter(
            torch.zeros(1, input_channels, channel_hidden, 1)
        )

        self.dropout = nn.Dropout(dropout)

        # Смешивание channel-aware признаков в размер baseline stem.
        # Здесь уже можно смешивать FrL/FrR/OcR_Hipp.
        self.mix = nn.Sequential(
            nn.Conv1d(input_channels * channel_hidden, hidden,
                      kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # x: (B, C, T)
        b, c, t = x.shape
        if c != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {c}"
            )

        # Применяем один shared Conv1d к каждому физическому каналу.
        z = x.reshape(b * c, 1, t)           # (B*C, 1, T)
        z = self.shared_conv(z)              # (B*C, Hc, T)
        z = self.act(z)
        z = z.reshape(b, c, self.channel_hidden, t)  # (B, C, Hc, T)

        # Маленькая per-channel поправка, без независимых фильтров.
        z = z * self.channel_gain + self.channel_bias

        z = z.reshape(b, c * self.channel_hidden, t) # (B, C*Hc, T)
        z = self.dropout(z)
        z = self.mix(z)                              # (B, hidden, T)
        return z


@register_model("RDSCBiGRUDetector_ChannelAdapterShared_v1")
class RDSCBiGRUDetector_ChannelAdapterShared_v1(nn.Module):
    """
    RDSCBiGRUDetector + маленький residual channel-aware adapter.

    Отличие от RDSCBiGRUDetector:
      - основной stem Conv1d(3 -> 32) сохранён;
      - параллельно добавлена маленькая shared-weight ветка,
        которая применяет один и тот же Conv1d(1 -> 4) к каждому
        физическому каналу;
      - результат adapter-ветки добавляется к baseline stem через
        learnable scalar adapter_scale.

    Цель:
      - сохранить устойчивый early-fusion baseline;
      - добавить channel-aware поправку без трёх независимых фильтробанков;
      - не использовать input normalization / GroupNorm.

    Вход:  (B, 3, 2000)
    Выход: (B, 2000) logits
    """
    def __init__(
        self,
        input_channels=3,
        hidden=32,
        channel_hidden=4,
        gru_hidden=32,
        gru_layers=1,
        dropout=0.15,
        adapter_dropout=0.05,
        adapter_scale_init=0.05,
    ):
        super().__init__()

        # 1. Успешный baseline stem из RDSCBiGRUDetector.
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, hidden,
                      kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        # 2. Маленькая residual channel-aware ветка.
        self.channel_adapter = SharedChannelAdapter1d(
            input_channels=input_channels,
            channel_hidden=channel_hidden,
            hidden=hidden,
            kernel_size=7,
            dropout=adapter_dropout,
        )

        # Не ставим 0.0, чтобы adapter сразу получал градиент.
        # 0.05 достаточно мало, чтобы не разрушать baseline stem на старте.
        self.adapter_scale = nn.Parameter(
            torch.tensor(float(adapter_scale_init))
        )
        self.combine_act = nn.ReLU(inplace=True)

        # 3. Shared RDSC trunk, как в успешном RDSCBiGRUDetector.
        self.rdsc_blocks = nn.Sequential(
            RDSCBlock(hidden, kernel_size=9, dropout=dropout),
            RDSCBlock(hidden, kernel_size=9, dropout=dropout),
            RDSCBlock(hidden, kernel_size=7, dropout=dropout),
        )

        # 4. Temporal smoother.
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # 5. Per-frame classifier.
        self.head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x):
        x_base = self.stem(x)                    # (B, hidden, T)
        x_adapter = self.channel_adapter(x)      # (B, hidden, T)

        x = x_base + self.adapter_scale * x_adapter
        x = self.combine_act(x)

        x = self.rdsc_blocks(x)                  # (B, hidden, T)
        x = x.permute(0, 2, 1)                   # (B, T, hidden)
        x, _ = self.gru(x)                       # (B, T, gru_hidden*2)
        x = self.head(x).squeeze(-1)             # (B, T)
        return x
