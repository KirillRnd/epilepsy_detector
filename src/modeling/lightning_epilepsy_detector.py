import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from typing import Optional
import torchmetrics

from .simple_cnn_detector import SimpleEEGDetector, ImprovedEEGDetector, MinimalEEGDetector, MinimalEEGDetector_v2, MinimalEEGDetector_ESN
       
class EpilepsyDetector_v2(pl.LightningModule):
    """
    PyTorch Lightning модуль для детектирования эпилепсии
    """
    
    def __init__(self, 
                 input_channels: int = 4,
                 window_length: int = 2000,
                 num_classes: int = 2,
                 dropout_rate: float = 0.5,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4,
                 use_improved_architecture: bool = False,
                 use_minimal_architecture: bool = False,
                 class_weights: Optional[list] = None):
        """
        Инициализация модели
        
        Параметры:
        input_channels (int): количество каналов ЭЭГ
        window_length (int): длина временного окна в отсчетах
        num_classes (int): количество классов
        dropout_rate (float): вероятность dropout
        learning_rate (float): скорость обучения
        weight_decay (float): коэффициент регуляризации L2
        use_improved_architecture (bool): использовать улучшенную архитектуру
        use_minimal_architecture (bool): использовать минимальную архитектуру
        class_weights (list): веса классов для балансировки
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Инициализация модели
        self.model = MinimalEEGDetector_v2()
        
        # Параметры обучения
        self.learning_rate = learning_rate
        # Убедимся, что weight_decay является числом с плавающей точкой
        self.weight_decay = float(weight_decay)
        self.threshold = 0.5
        # Веса классов
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None
        pos_weight = 100
        self.register_buffer("pos_weight", torch.tensor([pos_weight], dtype=torch.float32))
        
        # Хранилище для результатов валидации и тестирования
        # Метрики (агрегируют по всем точкам; будем flatten)
        self.train_acc = torchmetrics.classification.BinaryAccuracy(threshold=self.threshold)
        self.val_acc = torchmetrics.classification.BinaryAccuracy(threshold=self.threshold)
        self.test_acc = torchmetrics.classification.BinaryAccuracy(threshold=self.threshold)

        self.train_f1 = torchmetrics.classification.BinaryF1Score(threshold=self.threshold)
        self.val_f1 = torchmetrics.classification.BinaryF1Score(threshold=self.threshold)
        self.test_f1 = torchmetrics.classification.BinaryF1Score(threshold=self.threshold)

        self.train_precision = torchmetrics.classification.BinaryPrecision(threshold=self.threshold)
        self.val_precision = torchmetrics.classification.BinaryPrecision(threshold=self.threshold)
        self.test_precision = torchmetrics.classification.BinaryPrecision(threshold=self.threshold)

        self.train_recall = torchmetrics.classification.BinaryRecall(threshold=self.threshold)
        self.val_recall = torchmetrics.classification.BinaryRecall(threshold=self.threshold)
        self.test_recall = torchmetrics.classification.BinaryRecall(threshold=self.threshold)
    
    def forward(self, x):
        return self.model(x)
    
    @staticmethod
    def _flatten_time(logits: torch.Tensor, targets: torch.Tensor):
        """
        Приводим к (N,) где N = B*T для torchmetrics.
        """
        probs = torch.sigmoid(logits)
        return probs.reshape(-1), targets.reshape(-1).int()
    
    def configure_optimizers(self):
        """
        Настройка оптимизатора и планировщика
        
        Возвращает:
        dict: конфигурация оптимизатора и планировщика
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=float(self.weight_decay)
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            patience=5,
            factor=0.5
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def _compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, T) float
        targets: (B, T) float {0,1}
        """
        # Важно: targets должны быть float для BCE
        targets = targets.float()

        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)

        return loss_fn(logits, targets)
    
    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B,C,T), y: (B,T) 0/1
        logits = self(x)               # (B,T)
        loss = self._compute_loss(logits, y)

        probs_flat, y_flat = self._flatten_time(logits, y)

        self.train_acc.update(probs_flat, y_flat)
        self.train_f1.update(probs_flat, y_flat)
        self.train_precision.update(probs_flat, y_flat)
        self.train_recall.update(probs_flat, y_flat)

        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True, batch_size=x.size(0))
        self.log("train_acc", self.train_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_f1", self.train_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_precision", self.train_precision, prog_bar=False, on_step=False, on_epoch=True)
        self.log("train_recall", self.train_recall, prog_bar=False, on_step=False, on_epoch=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)

        probs_flat, y_flat = self._flatten_time(logits, y)

        self.val_acc.update(probs_flat, y_flat)
        self.val_f1.update(probs_flat, y_flat)
        self.val_precision.update(probs_flat, y_flat)
        self.val_recall.update(probs_flat, y_flat)

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("val_acc", self.val_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_f1", self.val_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_precision", self.val_precision, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_recall", self.val_recall, prog_bar=False, on_step=False, on_epoch=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)

        probs_flat, y_flat = self._flatten_time(logits, y)

        self.test_acc.update(probs_flat, y_flat)
        self.test_f1.update(probs_flat, y_flat)
        self.test_precision.update(probs_flat, y_flat)
        self.test_recall.update(probs_flat, y_flat)

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True, batch_size=x.size(0))
        self.log("test_acc", self.test_acc, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_f1", self.test_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_precision", self.test_precision, prog_bar=False, on_step=False, on_epoch=True)
        self.log("test_recall", self.test_recall, prog_bar=False, on_step=False, on_epoch=True)

        return loss
        
    def on_train_epoch_end(self):

        self.train_acc.reset()
        self.train_f1.reset()
        self.train_precision.reset()
        self.train_recall.reset()

    def on_validation_epoch_end(self):

        self.val_acc.reset()
        self.val_f1.reset()
        self.val_precision.reset()
        self.val_recall.reset()

    def on_test_epoch_end(self):
        self.test_acc.reset()
        self.test_f1.reset()
        self.test_precision.reset()
        self.test_recall.reset()