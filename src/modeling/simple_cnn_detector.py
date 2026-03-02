import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_registry import register_model

@register_model("minimal_v2")
class MinimalEEGDetector_v2(nn.Module):
    """
    Минимальная fully-conv 1D-CNN для детекции seizure по каждому семплу окна.
    Вход:  (B, C=3, T=2000)
    Выход: (B, T=2000) logits
    """

    def __init__(self, input_channels: int = 3, hidden: int = 16, dropout: float = 0.1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv: превращаем признаки в 1 логит на каждый момент времени
        self.head = nn.Conv1d(hidden // 2, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns logits: (B, T)
        """
        x = self.features(x)          # (B, H, T)
        x = self.head(x)              # (B, 1, T)
        x = x.squeeze(1)              # (B, T)
        return x

class ESNLayer(nn.Module):
    def __init__(
        self,
        input_size: int,        # C
        reservoir_size: int,    # H
        leaking_rate: float = 1.0,
        spectral_radius: float = 0.9,
        input_scale: float = 1.0,
        bias: bool = True,
        train_reservoir: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate

        # W_in: (H, C)
        self.W_in = nn.Parameter(
            torch.empty(reservoir_size, input_size),
            requires_grad=train_reservoir
        )
        # W_res: (H, H)
        self.W_res = nn.Parameter(
            torch.empty(reservoir_size, reservoir_size),
            requires_grad=train_reservoir
        )
        if bias:
            self.b = nn.Parameter(torch.zeros(reservoir_size),
                                  requires_grad=train_reservoir)
        else:
            self.register_parameter("b", None)

        self.input_scale = input_scale
        self.spectral_radius = spectral_radius

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # случайная инициализация
        nn.init.normal_(self.W_in, mean=0.0, std=self.input_scale)
        nn.init.normal_(self.W_res, mean=0.0, std=1.0)

        # нормируем на заданный spectral_radius
        # оцениваем спектральный радиус по максимальному сингулярному значению
        # (приближение, но на практике достаточно)
        u, s, v = torch.svd(self.W_res)
        if s[0] > 0:
            self.W_res.data *= self.spectral_radius / s[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: states: (B, H, T)
        """
        B, C, T = x.shape
        device = x.device

        # (B, T, C)
        x_t = x.permute(0, 2, 1)

        # начальное состояние резервуара
        h = torch.zeros(B, self.reservoir_size, device=device)

        states = []

        for t in range(T):
            u_t = x_t[:, t, :]  # (B, C)

            pre = F.linear(u_t, self.W_in) + F.linear(h, self.W_res)
            if self.b is not None:
                pre = pre + self.b

            pre = torch.tanh(pre)

            # leaky integration
            h = (1.0 - self.leaking_rate) * h + self.leaking_rate * pre

            states.append(h.unsqueeze(2))  # (B, H, 1)

        # (B, H, T)
        states = torch.cat(states, dim=2)
        return states

@register_model("ESN")
class MinimalEEGDetector_ESN(nn.Module):
    def __init__(self,
                 input_channels: int = 3,
                 esn_hidden: int = 32,
                 conv_hidden: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        self.esn = ESNLayer(
            input_size=input_channels,
            reservoir_size=esn_hidden,
            leaking_rate=0.3,
            spectral_radius=0.9,
            input_scale=0.5,
            bias=True,
            train_reservoir=False,  # сначала можно заморозить
        )

        hidden = conv_hidden
        self.features = nn.Sequential(
            nn.Conv1d(esn_hidden, hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv1d(hidden // 2, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.esn(x)          # (B, esn_hidden, T)
        x = self.features(x)     # (B, H, T)
        x = self.head(x)         # (B, 1, T)
        x = x.squeeze(1)         # (B, T)
        return x
