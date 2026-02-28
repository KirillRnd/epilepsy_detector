import torch
import numpy as np
from torch.utils.data import Sampler, WeightedRandomSampler
from collections import Counter
from typing import List, Tuple
import pytorch_lightning as pl

class BalancedSampler(Sampler):
    """
    Сэмплер для балансировки классов в батчах
    """
    
    def __init__(self, dataset, batch_size: int, target_ratio: float = 0.25):
        """
        Инициализация сбалансированного сэмплера
        
        Параметры:
        dataset: датасет
        batch_size (int): размер батча
        target_ratio (float): целевое соотношение minority класса (приступы)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.target_ratio = target_ratio
        
        # Получение меток из датасета
        labels = self._get_labels()
        
        # Вычисление весов для балансировки
        self.weights = self._compute_weights(labels)
    
    def _get_labels(self) -> List[int]:
        """
        Получение всех меток из датасета
        
        Возвращает:
        list: список меток
        """
        labels = []
        for i in range(len(self.dataset)):
            _, label = self.dataset[i]
            labels.append(label)
        return labels
    
    def _compute_weights(self, labels: List[int]) -> List[float]:
        """
        Вычисление весов для балансировки классов
        
        Параметры:
        labels (list): список меток
        
        Возвращает:
        list: список весов
        """
        # Подсчет количества примеров каждого класса
        label_counts = Counter(labels)
        total_samples = len(labels)
        
        # Вычисление весов
        weights = []
        for label in labels:
            # Вес обратно пропорционален частоте класса
            weight = total_samples / (len(label_counts) * label_counts[label])
            weights.append(weight)
        
        return weights
    
    def __iter__(self):
        """
        Итератор по индексам с учетом весов
        """
        sampler = WeightedRandomSampler(
            weights=self.weights,
            num_samples=len(self.weights),
            replacement=True
        )
        return iter(sampler)
    
    def __len__(self):
        """
        Длина сэмплера
        """
        return len(self.weights)

class EEGAugmentation:
    """
    Класс для аугментации данных ЭЭГ
    """
    
    def __init__(self, noise_level: float = 0.01, time_shift_range: int = 10):
        """
        Инициализация аугментации
        
        Параметры:
        noise_level (float): уровень шума
        time_shift_range (int): диапазон сдвига по времени
        """
        self.noise_level = noise_level
        self.time_shift_range = time_shift_range
    
    def add_noise(self, data: torch.Tensor) -> torch.Tensor:
        """
        Добавление шума к данным
        
        Параметры:
        data (torch.Tensor): входные данные
        
        Возвращает:
        torch.Tensor: данные с добавленным шумом
        """
        noise = torch.randn_like(data) * self.noise_level * torch.std(data)
        return data + noise
    
    def time_shift(self, data: torch.Tensor) -> torch.Tensor:
        """
        Сдвиг данных по времени
        
        Параметры:
        data (torch.Tensor): входные данные
        
        Возвращает:
        torch.Tensor: данные со сдвигом
        """
        shift = torch.randint(-self.time_shift_range, self.time_shift_range + 1, (1,)).item()
        return torch.roll(data, shifts=shift, dims=1)
    
    def amplitude_scaling(self, data: torch.Tensor, 
                        scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
        """
        Изменение амплитуды данных
        
        Параметры:
        data (torch.Tensor): входные данные
        scale_range (tuple): диапазон масштабирования
        
        Возвращает:
        torch.Tensor: данные с измененной амплитудой
        """
        scale = torch.FloatTensor(1).uniform_(scale_range[0], scale_range[1]).item()
        return data * scale
    
    def augment(self, data: torch.Tensor) -> torch.Tensor:
        """
        Применение случайной аугментации
        
        Параметры:
        data (torch.Tensor): входные данные
        
        Возвращает:
        torch.Tensor: аугментированные данные
        """
        # Случайный выбор аугментации
        augmentations = [
            lambda x: x,  # Без изменений
            self.add_noise,
            self.time_shift,
            self.amplitude_scaling
        ]
        
        # Выбор случайной аугментации
        aug_func = np.random.choice(augmentations)
        return aug_func(data)

def compute_class_weights(dataset) -> torch.Tensor:
    """
    Вычисление весов классов для балансировки
    
    Параметры:
    dataset: датасет
    
    Возвращает:
    torch.Tensor: веса классов
    """
    # Получение меток
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    # Подсчет количества примеров каждого класса
    label_counts = Counter(labels)
    total_samples = len(labels)
    num_classes = len(label_counts)
    
    # Вычисление весов
    weights = torch.zeros(num_classes)
    for class_label, count in label_counts.items():
        weights[class_label] = total_samples / (num_classes * count)
    
    return weights