import torch
import torch.nn as nn
from .model_registry import register_model


class ChannelSEBlock1d(nn.Module):
    """Squeeze-and-Excitation for one physical-channel branch."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        bottleneck = max(1, channels // reduction)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        b, c, _ = x.shape
        s = self.squeeze(x).view(b, c)
        s = self.excitation(s).view(b, c, 1)
        return x * s


class ChannelRDSCBlock(nn.Module):
    """RDSC block inside one physical-channel branch."""

    def __init__(self, channels: int, kernel_size: int = 9, dropout: float = 0.15):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            # Depthwise temporal filtering inside the branch.
            nn.Conv1d(channels, channels, kernel_size,
                      padding=pad, groups=channels, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),

            # Pointwise mixing only between hidden features of this same
            # physical-channel branch, not between EEG channels.
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.se = ChannelSEBlock1d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out = self.se(out)
        return out + x


class ChannelStage(nn.Module):
    """
    One RDSC block per physical EEG channel.

    Input/output: (B, P, H, T), where:
      P = number of physical channels,
      H = hidden features per physical channel.
    """

    def __init__(self, num_phys_channels: int, hidden_per_channel: int,
                 kernel_size: int = 9, dropout: float = 0.15):
        super().__init__()
        self.branches = nn.ModuleList([
            ChannelRDSCBlock(hidden_per_channel, kernel_size=kernel_size, dropout=dropout)
            for _ in range(num_phys_channels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, H, T)
        ys = [branch(x[:, i, :, :]) for i, branch in enumerate(self.branches)]
        return torch.stack(ys, dim=1)  # (B, P, H, T)


class WeakChannelMix(nn.Module):
    """
    Weak residual mixing between physical channels.

    Mixing is feature-wise: for each hidden feature h, only physical channels
    are mixed with each other. Hidden features are not globally mixed.
    Self-channel terms are masked out, so the residual identity path is explicit.
    """

    def __init__(self, num_phys_channels: int, hidden_per_channel: int,
                 alpha_init: float = 0.05):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.mix = nn.Parameter(torch.zeros(
            hidden_per_channel, num_phys_channels, num_phys_channels
        ))

        offdiag = torch.ones(hidden_per_channel, num_phys_channels, num_phys_channels)
        eye = torch.eye(num_phys_channels).unsqueeze(0)
        offdiag = offdiag - eye
        self.register_buffer("offdiag_mask", offdiag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, P, H, T)
        weight = self.mix * self.offdiag_mask
        mixed = torch.einsum("hoq,bqht->boht", weight, x)
        return x + self.alpha * mixed


@register_model("StagedChannelRDSCBiGRUDetector_v2")
class StagedChannelRDSCBiGRUDetector_v2(nn.Module):
    """
    Compact channel-specific staged RDSC + weak channel mixing + BiGRU.

    Difference from the original RDSCBiGRUDetector:
      - no raw input normalization;
      - physical EEG channels are kept isolated in the stem and stage blocks;
      - each stage has one RDSC block per physical channel;
      - two weak residual cross-channel mixes are inserted between stages;
      - strong channel fusion is delayed until the final 1x1 conv before BiGRU.

    Difference from StagedChannelRDSCBiGRUDetector_v1:
      - v1 accidentally used three RDSC blocks per channel in each stage;
      - v2 uses one RDSC block per channel in each stage.

    Defaults are intentionally compact and descriptor-friendly:
      input:  (B, 3, 2000)
      output: (B, 2000) logits
    """

    def __init__(self, input_channels: int = 3, hidden_per_channel: int = 6,
                 final_hidden: int = 32, gru_hidden: int = 32,
                 gru_layers: int = 1, dropout: float = 0.15):
        super().__init__()
        if input_channels != 3:
            raise ValueError(
                "StagedChannelRDSCBiGRUDetector_v2 is currently designed for "
                f"exactly 3 physical EEG channels, got input_channels={input_channels}"
            )

        self.input_channels = input_channels
        self.hidden_per_channel = hidden_per_channel
        self.final_hidden = final_hidden

        # Independent stems for FrL / FrR / OcR_Hipp.
        self.stems = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden_per_channel, kernel_size=7,
                          padding=3, bias=False),
                nn.BatchNorm1d(hidden_per_channel),
                nn.ReLU(inplace=True),
            )
            for _ in range(input_channels)
        ])

        # Three channel-isolated stages: one block per physical channel per stage.
        self.stage1 = ChannelStage(input_channels, hidden_per_channel,
                                   kernel_size=9, dropout=dropout)
        self.weak_mix1 = WeakChannelMix(input_channels, hidden_per_channel,
                                        alpha_init=0.05)

        self.stage2 = ChannelStage(input_channels, hidden_per_channel,
                                   kernel_size=9, dropout=dropout)
        self.weak_mix2 = WeakChannelMix(input_channels, hidden_per_channel,
                                        alpha_init=0.05)

        self.stage3 = ChannelStage(input_channels, hidden_per_channel,
                                   kernel_size=7, dropout=dropout)

        # Strong final fusion: only after the staged channel-specific extractor.
        self.final_mix = nn.Sequential(
            nn.Conv1d(input_channels * hidden_per_channel, final_hidden,
                      kernel_size=1, bias=False),
            nn.BatchNorm1d(final_hidden),
            nn.ReLU(inplace=True),
        )

        self.gru = nn.GRU(
            input_size=final_hidden,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )

        self.head = nn.Linear(gru_hidden * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, T)
        if x.dim() != 3 or x.size(1) != self.input_channels:
            raise ValueError(
                "Expected input shape (B, 3, T), got " + str(tuple(x.shape))
            )

        xs = [stem(x[:, i:i + 1, :]) for i, stem in enumerate(self.stems)]
        x = torch.stack(xs, dim=1)  # (B, 3, H, T)

        x = self.stage1(x)
        x = self.weak_mix1(x)
        x = self.stage2(x)
        x = self.weak_mix2(x)
        x = self.stage3(x)

        b, p, h, t = x.shape
        x = x.contiguous().view(b, p * h, t)  # (B, 3*H, T)
        x = self.final_mix(x)                 # (B, final_hidden, T)

        x = x.permute(0, 2, 1)                # (B, T, final_hidden)
        x, _ = self.gru(x)                    # (B, T, 2*gru_hidden)
        x = self.head(x).squeeze(-1)          # (B, T)
        return x
