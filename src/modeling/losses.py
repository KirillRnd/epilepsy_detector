import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss для бинарной покадровой классификации.
    Снижает вклад «лёгких» примеров и фокусирует обучение
    на трудных (границы приступов, переходные зоны).

    Параметры:
        alpha: вес позитивного класса (приступ). При alpha=0.75
               приступы получают в 3 раза больший вес, чем норма.
        gamma: степень фокусировки. gamma=0 → обычный BCE.
               gamma=2 — стандартное значение.
    """

    def __init__(self, alpha: float = 0.95, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, T) — сырые логиты модели
        targets: (B, T) — метки 0/1 (или soft targets от Mixup/Label Smoothing)
        """
        # Стандартный BCE без редукции
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Вероятность правильного класса
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Весовой коэффициент alpha для каждого класса
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Focal-множитель: (1 - p_t)^gamma
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()
