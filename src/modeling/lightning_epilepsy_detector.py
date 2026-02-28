import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from typing import Optional

from .simple_cnn_detector import SimpleEEGDetector, ImprovedEEGDetector, MinimalEEGDetector

class EpilepsyDetector(pl.LightningModule):
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
        if use_minimal_architecture:
            self.model = MinimalEEGDetector(
                input_channels=input_channels,
                window_length=window_length,
                num_classes=num_classes
            )
        elif use_improved_architecture:
            self.model = ImprovedEEGDetector(
                input_channels=input_channels,
                window_length=window_length,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
        else:
            self.model = SimpleEEGDetector(
                input_channels=input_channels,
                window_length=window_length,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
        
        # Параметры обучения
        self.learning_rate = learning_rate
        # Убедимся, что weight_decay является числом с плавающей точкой
        self.weight_decay = float(weight_decay)
        
        # Веса классов
        if class_weights is not None:
            self.class_weights = torch.FloatTensor(class_weights)
        else:
            self.class_weights = None
        
        # Хранилище для результатов валидации и тестирования
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Параметры:
        x (torch.Tensor): входные данные
        
        Возвращает:
        torch.Tensor: выходные логиты
        """
        return self.model(x)
    
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
    
    def _compute_loss(self, outputs, targets):
        """
        Вычисление функции потерь
        
        Параметры:
        outputs (torch.Tensor): выходы модели
        targets (torch.Tensor): истинные метки
        
        Возвращает:
        torch.Tensor: значение функции потерь
        """
        if self.class_weights is not None:
            # Убедимся, что веса на том же устройстве
            class_weights = self.class_weights.to(outputs.device)
            return F.cross_entropy(outputs, targets, weight=class_weights)
        else:
            return F.cross_entropy(outputs, targets)
    
    def training_step(self, batch, batch_idx):
        """
        Шаг обучения
        
        Параметры:
        batch: батч данных
        batch_idx (int): индекс батча
        
        Возвращает:
        dict: словарь с результатами шага
        """
        x, y = batch
        outputs = self(x)
        loss = self._compute_loss(outputs, y)
        
        # Вычисление метрик
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Логирование
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return {'loss': loss, 'acc': acc}
    
    def validation_step(self, batch, batch_idx):
        """
        Шаг валидации
        
        Параметры:
        batch: батч данных
        batch_idx (int): индекс батча
        
        Возвращает:
        dict: словарь с результатами шага
        """
        x, y = batch
        outputs = self(x)
        loss = self._compute_loss(outputs, y)
        
        # Вычисление метрик
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Логирование
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        # Сохраняем результаты для обработки в конце эпохи
        self.validation_step_outputs.append({'preds': preds, 'targets': y})
        
        return {'val_loss': loss, 'val_acc': acc, 'preds': preds, 'targets': y}
    
    def on_validation_epoch_end(self):
        """
        Обработка результатов валидации в конце эпохи
        """
        # Объединение предсказаний и целевых значений
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.validation_step_outputs])
        
        # Очистка хранилища
        self.validation_step_outputs.clear()
        
        # Вычисление метрик
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        
        # Метрики классификации
        precision = precision_score(targets_np, preds_np, zero_division=0)
        recall = recall_score(targets_np, preds_np, zero_division=0)
        f1 = f1_score(targets_np, preds_np, zero_division=0)
        
        # Sensitivity и Specificity
        cm = np.zeros((2, 2))
        if len(np.unique(targets_np)) > 1:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(targets_np, preds_np, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                sensitivity = 0
                specificity = 0
        else:
            sensitivity = 0
            specificity = 0
        
        # Логирование метрик
        self.log('val_precision', precision, on_epoch=True, logger=True)
        self.log('val_recall', recall, on_epoch=True, logger=True)
        self.log('val_f1', f1, on_epoch=True, logger=True)
        self.log('val_sensitivity', sensitivity, on_epoch=True, logger=True)
        self.log('val_specificity', specificity, on_epoch=True, logger=True)
    
    def test_step(self, batch, batch_idx):
        """
        Шаг тестирования
        
        Параметры:
        batch: батч данных
        batch_idx (int): индекс батча
        
        Возвращает:
        dict: словарь с результатами шага
        """
        x, y = batch
        outputs = self(x)
        loss = self._compute_loss(outputs, y)
        
        # Вычисление метрик
        preds = torch.argmax(outputs, dim=1)
        acc = accuracy_score(y.cpu(), preds.cpu())
        
        # Логирование
        self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('test_acc', acc, on_step=False, on_epoch=True, logger=True)
        
        # Возвращаем предсказания и вероятности для вычисления ROC AUC
        probs = F.softmax(outputs, dim=1)
        
        # Сохраняем результаты для обработки в конце эпохи
        self.test_step_outputs.append({'preds': preds, 'targets': y, 'probs': probs})
        
        return {
            'test_loss': loss,
            'test_acc': acc,
            'preds': preds,
            'targets': y,
            'probs': probs
        }
    
    def on_test_epoch_end(self):
        """
        Обработка результатов тестирования в конце эпохи
        """
        # Объединение предсказаний, целевых значений и вероятностей
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs])
        all_targets = torch.cat([x['targets'] for x in self.test_step_outputs])
        all_probs = torch.cat([x['probs'] for x in self.test_step_outputs])
        
        # Очистка хранилища
        self.test_step_outputs.clear()
        
        # Вычисление метрик
        preds_np = all_preds.cpu().numpy()
        targets_np = all_targets.cpu().numpy()
        probs_np = all_probs.cpu().numpy()
        
        # Базовые метрики
        acc = accuracy_score(targets_np, preds_np)
        precision = precision_score(targets_np, preds_np, zero_division=0)
        recall = recall_score(targets_np, preds_np, zero_division=0)
        f1 = f1_score(targets_np, preds_np, zero_division=0)
        
        # ROC AUC
        try:
            if len(np.unique(targets_np)) > 1 and probs_np.shape[1] == 2:
                auc = roc_auc_score(targets_np, probs_np[:, 1])
            else:
                auc = 0.0
        except:
            auc = 0.0
        
        # Sensitivity и Specificity
        cm = np.zeros((2, 2))
        if len(np.unique(targets_np)) > 1:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(targets_np, preds_np, labels=[0, 1])
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                sensitivity = 0
                specificity = 0
        else:
            sensitivity = 0
            specificity = 0
        
        # Логирование метрик
        self.log('test_accuracy', acc, on_epoch=True, logger=True)
        self.log('test_precision', precision, on_epoch=True, logger=True)
        self.log('test_recall', recall, on_epoch=True, logger=True)
        self.log('test_f1', f1, on_epoch=True, logger=True)
        self.log('test_auc', auc, on_epoch=True, logger=True)
        self.log('test_sensitivity', sensitivity, on_epoch=True, logger=True)
        self.log('test_specificity', specificity, on_epoch=True, logger=True)
        
        # Вывод результатов
        print(f"\nРезультаты тестирования:")
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")