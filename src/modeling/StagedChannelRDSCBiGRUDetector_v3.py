import torch
import torch.nn as nn
from .model_registry import register_model


class GroupedSEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation без смешивания физических каналов.

    Вход:  (B, num_physical_channels * channels_per_group, T)
    Выход: тот же shape.

    В отличие от обычного SEBlock, здесь MLP реализован через grouped 1x1 Conv:
    каждый физический канал получает свой SE-гейт, но не смотрит на другие
    физические каналы.
    """
    def __init__(self, num_physical_channels: int, channels_per_group: int, reduction: int = 4):
        super().__init__()
        total_channels = num_physical_channels * channels_per_group
        reduced_per_group = max(1, channels_per_group // reduction)

        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Conv1d(
                total_channels,
                num_physical_channels * reduced_per_group,
                kernel_size=1,
                groups=num_physical_channels,
                bias=False,
            ),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                num_physical_channels * reduced_per_group,
                total_channels,
                kernel_size=1,
                groups=num_physical_channels,
                bias=False,
            ),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.squeeze(x)       # (B, C_total, 1)
        s = self.excitation(s)    # (B, C_total, 1)
        return x * s


class ChannelIsolatedRDSCBlock(nn.Module):
    """
    RDSC-блок, который не смешивает физические каналы.

    Каналы устроены как:
        [FrL_features, FrR_features, OcR/Hipp_features]

    Depthwise conv работает по каждому hidden feature отдельно.
    Pointwise conv имеет groups=num_physical_channels, поэтому смешивает признаки
    только внутри одного физического канала, но не между FrL/FrR/OcR_Hipp.
    SE-гейт тоже grouped и не смешивает физические каналы.
    """
    def __init__(
        self,
        num_physical_channels: int,
        channels_per_group: int,
        kernel_size: int = 9,
        dropout: float = 0.15,
    ):
        super().__init__()
        total_channels = num_physical_channels * channels_per_group
        pad = kernel_size // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                total_channels,
                total_channels,
                kernel_size,
                padding=pad,
                groups=total_channels,
                bias=False,
            ),
            nn.BatchNorm1d(total_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                total_channels,
                total_channels,
                kernel_size=1,
                groups=num_physical_channels,
                bias=False,
            ),
            nn.BatchNorm1d(total_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.se = GroupedSEBlock1d(
            num_physical_channels=num_physical_channels,
            channels_per_group=channels_per_group,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        return out + x


class SEBlock1d(nn.Module):
    """Обычный SEBlock для смешанного hidden-представления."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1)
        return x * s


class RDSCBlock(nn.Module):
    """Обычный смешанный Residual Depthwise Separable Conv + SE attention."""
    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.15):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=pad,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.se = SEBlock1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        return out + x


@register_model("StagedChannelRDSCBiGRUDetector_v3")
class StagedChannelRDSCBiGRUDetector_v3(nn.Module):
    """
    Channel-aware RDSC + BiGRU.

    Идея v3:
      - не нормализуем сырой вход;
      - сначала даём каждому физическому каналу свой короткий extractor;
      - затем рано смешиваем каналы в общее hidden-представление;
      - дальше почти повторяем успешный RDSCBiGRUDetector trunk.

    Отличие от v2:
      - нет долгой изоляции каналов;
      - нет нескольких weak-mix этапов;
      - после короткого channel-specific блока идёт ранний strong 1x1 mix;
      - дальше 3 обычных RDSC-блока на смешанном представлении.

    Вход:  (B, 3, 2000)
    Выход: (B, 2000) logits
    """
    def __init__(
        self,
        input_channels: int = 3,
        channel_hidden: int = 8,
        hidden: int = 32,
        gru_hidden: int = 32,
        gru_layers: int = 1,
        dropout: float = 0.15,
    ):
        super().__init__()

        # 1. Channel-specific stem.
        # groups=input_channels означает: каждый физический канал обрабатывается
        # своим набором фильтров, без раннего смешивания FrL/FrR/OcR_Hipp.
        self.channel_stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                input_channels * channel_hidden,
                kernel_size=7,
                padding=3,
                groups=input_channels,
                bias=False,
            ),
            nn.BatchNorm1d(input_channels * channel_hidden),
            nn.ReLU(inplace=True),
        )

        # 2. Один isolated RDSC-блок: признаки смешиваются только внутри
        # каждого физического канала.
        self.channel_block = ChannelIsolatedRDSCBlock(
            num_physical_channels=input_channels,
            channels_per_group=channel_hidden,
            kernel_size=9,
            dropout=dropout,
        )

        # 3. Раннее controlled/strong смешивание физических каналов.
        # После этого возвращаемся к удачному shared RDSC-trunk.
        self.early_mix = nn.Sequential(
            nn.Conv1d(
                input_channels * channel_hidden,
                hidden,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
        )

        # 4. Shared RDSC trunk, близко к RDSCBiGRUDetector.
        self.rdsc_blocks = nn.Sequential(
            RDSCBlock(hidden, kernel_size=9, dropout=dropout),
            RDSCBlock(hidden, kernel_size=9, dropout=dropout),
            RDSCBlock(hidden, kernel_size=7, dropout=dropout),
        )

        # 5. Temporal smoother.
        self.gru = nn.GRU(
            input_size=hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        # 6. Per-frame classifier.
        self.head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_stem(x)      # (B, 3 * channel_hidden, T)
        x = self.channel_block(x)     # still channel-isolated
        x = self.early_mix(x)         # (B, hidden, T), channels mixed
        x = self.rdsc_blocks(x)       # shared mixed RDSC trunk
        x = x.permute(0, 2, 1)        # (B, T, hidden)
        x, _ = self.gru(x)            # (B, T, gru_hidden * 2)
        x = self.head(x).squeeze(-1)  # (B, T)
        return x
