import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class EEGAugmentor:
    """
    Набор аугментаций для ЭЭГ-сигналов с покадровыми метками.
    Каждый метод принимает (signal, target) и возвращает (signal, target).
    signal: (C, T), target: (T,)
    """

    def __init__(
        self,
        p_noise: float = 0.5,
        noise_std_range: Tuple[float, float] = (0.01, 0.1),
        p_scale: float = 0.5,
        scale_range: Tuple[float, float] = (0.7, 1.3),
        p_time_shift: float = 0.3,
        max_shift_samples: int = 80,  # ±200 мс при 400 Гц
        p_channel_dropout: float = 0.2,
        p_mixup: float = 0.3,
        mixup_alpha: float = 0.3,
        p_cutmix: float = 0.2,
        cutmix_min_len: int = 200,    # мин. длина вырезаемого участка
        cutmix_max_len: int = 800,    # макс. длина вырезаемого участка
        label_smooth_samples: int = 40,  # ±100 мс при 400 Гц
    ):
        self.p_noise = p_noise
        self.noise_std_range = noise_std_range
        self.p_scale = p_scale
        self.scale_range = scale_range
        self.p_time_shift = p_time_shift
        self.max_shift_samples = max_shift_samples
        self.p_channel_dropout = p_channel_dropout
        self.p_mixup = p_mixup
        self.mixup_alpha = mixup_alpha
        self.p_cutmix = p_cutmix
        self.cutmix_min_len = cutmix_min_len
        self.cutmix_max_len = cutmix_max_len
        self.label_smooth_samples = label_smooth_samples

    # ------------------------------------------------------------------ #
    #  Базовые преобразования сигнала
    # ------------------------------------------------------------------ #

    def add_gaussian_noise(self, signal: torch.Tensor) -> torch.Tensor:
        """Добавление гауссова шума, масштабированного по std каждого канала."""
        if np.random.random() > self.p_noise:
            return signal
        std_factor = np.random.uniform(*self.noise_std_range)
        channel_std = signal.std(dim=1, keepdim=True).clamp(min=1e-8)
        noise = torch.randn_like(signal) * channel_std * std_factor
        return signal + noise

    def amplitude_scaling(self, signal: torch.Tensor) -> torch.Tensor:
        """Случайное масштабирование амплитуды (одинаково для всех каналов)."""
        if np.random.random() > self.p_scale:
            return signal
        scale = np.random.uniform(*self.scale_range)
        return signal * scale

    def time_shift(
        self, signal: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Циклический сдвиг по времени (сигнал + метки синхронно)."""
        if np.random.random() > self.p_time_shift:
            return signal, target
        shift = np.random.randint(-self.max_shift_samples, self.max_shift_samples + 1)
        signal = torch.roll(signal, shifts=shift, dims=1)
        target = torch.roll(target, shifts=shift, dims=0)
        return signal, target

    def channel_dropout(self, signal: torch.Tensor) -> torch.Tensor:
        """Случайное обнуление одного из каналов."""
        if np.random.random() > self.p_channel_dropout:
            return signal
        ch = np.random.randint(0, signal.shape[0])
        signal = signal.clone()
        signal[ch] = 0.0
        return signal

    # ------------------------------------------------------------------ #
    #  Label Smoothing на границах приступов
    # ------------------------------------------------------------------ #

    def smooth_boundaries(self, target: torch.Tensor) -> torch.Tensor:
        """
        Gaussian ramp на границах приступов: метки плавно переходят 0→1 и 1→0
        за label_smooth_samples отсчётов (±100 мс по умолчанию).
        """
        if self.label_smooth_samples <= 0:
            return target
        target = target.clone().float()
        T = target.shape[0]
        half = self.label_smooth_samples

        # Находим фронты (0→1) и спады (1→0)
        diff = target[1:] - target[:-1]  # (T-1,)
        onsets = (diff > 0.5).nonzero(as_tuple=True)[0] + 1   # индексы начала приступа
        offsets = (diff < -0.5).nonzero(as_tuple=True)[0] + 1  # индексы конца приступа

        for idx in onsets:
            start = max(0, idx.item() - half)
            end = idx.item()
            if end > start:
                ramp = torch.linspace(0.0, 1.0, end - start)
                target[start:end] = ramp

        for idx in offsets:
            start = idx.item()
            end = min(T, idx.item() + half)
            if end > start:
                ramp = torch.linspace(1.0, 0.0, end - start)
                target[start:end] = ramp

        return target

    # ------------------------------------------------------------------ #
    #  Применение всех аугментаций к одному примеру
    # ------------------------------------------------------------------ #

    def __call__(
        self, signal: torch.Tensor, target: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Применяет цепочку аугментаций к одному примеру.
        signal: (C, T), target: (T,)
        """
        signal = self.add_gaussian_noise(signal)
        signal = self.amplitude_scaling(signal)
        signal, target = self.time_shift(signal, target)
        signal = self.channel_dropout(signal)
        target = self.smooth_boundaries(target)
        return signal, target


class MixupCutMixCollator:
    """
    Collate function для DataLoader, применяющая Mixup и CutMix на уровне батча.
    Эти аугментации смешивают ПАРЫ примеров, поэтому работают на уровне батча.
    """

    def __init__(
        self,
        p_mixup: float = 0.3,
        mixup_alpha: float = 0.3,
        p_cutmix: float = 0.2,
        cutmix_min_len: int = 200,
        cutmix_max_len: int = 800,
    ):
        self.p_mixup = p_mixup
        self.mixup_alpha = mixup_alpha
        self.p_cutmix = p_cutmix
        self.cutmix_min_len = cutmix_min_len
        self.cutmix_max_len = cutmix_max_len

    def __call__(self, batch):
        signals, targets = zip(*batch)
        signals = torch.stack(signals)  # (B, C, T)
        targets = torch.stack(targets)  # (B, T)
        B = signals.shape[0]

        r = np.random.random()
        if r < self.p_mixup:
            signals, targets = self._mixup(signals, targets)
        elif r < self.p_mixup + self.p_cutmix:
            signals, targets = self._cutmix(signals, targets)

        return signals, targets

    def _mixup(self, signals, targets):
        """Mixup: линейная комбинация пар примеров."""
        B = signals.shape[0]
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        perm = torch.randperm(B)
        signals = lam * signals + (1 - lam) * signals[perm]
        targets = lam * targets + (1 - lam) * targets[perm]
        return signals, targets

    def _cutmix(self, signals, targets):
        """CutMix: замена случайного временного отрезка из другого примера."""
        B, C, T = signals.shape
        perm = torch.randperm(B)
        cut_len = np.random.randint(self.cutmix_min_len, self.cutmix_max_len + 1)
        start = np.random.randint(0, T - cut_len)
        end = start + cut_len

        signals = signals.clone()
        targets = targets.clone()
        signals[:, :, start:end] = signals[perm, :, start:end]
        targets[:, start:end] = targets[perm, start:end]
        return signals, targets
