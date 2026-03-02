import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from typing import Optional, List
from sklearn.model_selection import train_test_split
import numpy as np

from .epilepsy_dataset import EpilepsyDataset, EpilepsyDataset_v2

class EpilepsyDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule для загрузки данных эпилепсии
    """
    
    def __init__(self, 
                 data_dir: str = "data/processed",
                 batch_size: int = 32,
                 window_length: int = 2000,
                 overlap: float = 0.5,
                 train_animal_ratio: float = 0.7,
                 val_animal_ratio: float = 0.15,
                 train_animals: list = None,
                 val_animals: list = None,
                 test_animals: list = None,
                 seed: int = 42):
        """
        Инициализация DataModule
        
        Параметры:
        data_dir (str): директория с предобработанными данными
        batch_size (int): размер батча
        window_length (int): длина окна в отсчетах
        overlap (float): перекрытие окон
        train_animal_ratio (float): доля животных для обучения
        val_animal_ratio (float): доля животных для валидации
        seed (int): seed для воспроизводимости
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.window_length = window_length
        self.overlap = overlap
        self.train_animal_ratio = train_animal_ratio
        self.val_animal_ratio = val_animal_ratio
        self.seed = seed
        
        # Сохраняем списки животных для жёсткого разбиения (если заданы)
        self.train_animals = train_animals
        self.val_animals = val_animals
        self.test_animals = test_animals
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def prepare_data(self):
        """
        Подготовка данных (вызывается только на одном GPU при распределенном обучении)
        """
        # Проверка существования директории с данными
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Директория с данными не найдена: {self.data_dir}")
    
    def setup(self, stage: Optional[str] = None):
        """
        Настройка датасетов для обучения, валидации и тестирования
        
        Параметры:
        stage (str): этап обучения ("fit", "test" или None)
        """
        # Загрузка информации о сегментах
        segments_dfs = []
        animal_ids = []
        
        # Проход по всем поддиректориям с данными
        for animal_dir in self.data_dir.iterdir():
            if animal_dir.is_dir():
                animal_id = animal_dir.name
                animal_ids.append(animal_id)
                
                # Проход по всем сессиям животного
                for session_dir in animal_dir.iterdir():
                    if session_dir.is_dir():
                        session_id = session_dir.name
                        segments_file = session_dir / "segments_info.csv"
                        
                        if segments_file.exists():
                            segments_df = pd.read_csv(segments_file)
                            segments_dfs.append(segments_df)
        
        # Объединение всех данных
        if segments_dfs:
            all_segments_df = pd.concat(segments_dfs, ignore_index=True)
        else:
            raise FileNotFoundError("Не найдены файлы с информацией о сегментах")
        
        # Разделение животных на train/val/test
        unique_animals = list(set(animal_ids))
        
        # Проверка, используется ли жёсткое разбиение
        if self.train_animals is not None and self.val_animals is not None and self.test_animals is not None:
            # Используем жёсткое разбиение
            train_animals = self.train_animals
            val_animals = self.val_animals
            test_animals = self.test_animals
            
            # Проверка, что все животные существуют
            all_specified_animals = set(train_animals + val_animals + test_animals)
            missing_animals = all_specified_animals - set(unique_animals)
            if missing_animals:
                raise ValueError(f"Следующие животные не найдены в данных: {missing_animals}")
        else:
            # Используем случайное разбиение
            np.random.seed(self.seed)
            np.random.shuffle(unique_animals)
            
            n_total = len(unique_animals)
            n_train = int(n_total * self.train_animal_ratio)
            n_val = int(n_total * self.val_animal_ratio)
            
            # Убедимся, что каждый сплит имеет хотя бы одно животное, если это возможно
            if n_total >= 3:  # Если у нас 3 или больше животных
                # Корректируем размеры, если какой-то сплит пустой
                if n_train == 0 and self.train_animal_ratio > 0:
                    n_train = 1
                if n_val == 0 and self.val_animal_ratio > 0:
                    n_val = 1
                
                # Убедимся, что сумма не превышает общее количество
                if n_train + n_val >= n_total:
                    # Распределяем животных пропорционально
                    n_train = max(1, int(n_total * self.train_animal_ratio))
                    n_val = max(1, min(n_total - n_train - 1, int(n_total * self.val_animal_ratio)))
                    # Если после коррекции валидация все еще пустая, но ratio > 0, даем ей 1
                    if n_val == 0 and self.val_animal_ratio > 0 and n_total > n_train + 1:
                        n_val = 1
            
            # Убедимся, что индексы не выходят за пределы массива
            n_train = min(n_train, n_total)
            remaining_for_val = max(0, n_total - n_train)
            n_val = min(n_val, remaining_for_val)
            
            # Убедимся, что у нас есть хотя бы по одному животному в каждом сплите, если возможно
            if n_total >= 3:
                if n_train == 0 and self.train_animal_ratio > 0:
                    n_train = 1
                if n_val == 0 and self.val_animal_ratio > 0 and remaining_for_val > 0:
                    n_val = 1
                if n_total - n_train - n_val == 0 and self.val_animal_ratio > 0 and n_val > 1:
                    n_val = max(1, n_val - 1)
            
            train_animals = unique_animals[:n_train]
            val_animals = unique_animals[n_train:n_train+n_val]
            test_animals = unique_animals[n_train+n_val:]
        
        # Фильтрация сегментов по животным
        train_segments = all_segments_df[all_segments_df['animal_id'].isin(train_animals)]
        val_segments = all_segments_df[all_segments_df['animal_id'].isin(val_animals)]
        test_segments = all_segments_df[all_segments_df['animal_id'].isin(test_animals)]
        
        # Создание датасетов
        if stage == "fit" or stage is None:
            self.train_dataset = EpilepsyDataset_v2(
                data_dir=str(self.data_dir),
                segments_df=train_segments,
                window_length=self.window_length,
                overlap=self.overlap
            )
            
            self.val_dataset = EpilepsyDataset_v2(
                data_dir=str(self.data_dir),
                segments_df=val_segments,
                window_length=self.window_length,
                overlap=self.overlap
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = EpilepsyDataset_v2(
                data_dir=str(self.data_dir),
                segments_df=test_segments,
                window_length=self.window_length,
                overlap=self.overlap
            )
        
        # Вывод разбиения животных на сеты
        print(f"Train animals: {train_animals}")
        print(f"Validation animals: {val_animals}")
        print(f"Test animals: {test_animals}")
    
    def train_dataloader(self) -> DataLoader:
        """
        Создание загрузчика обучающих данных
        """
        if self.train_dataset is None:
            raise RuntimeError("Датасет не инициализирован. Вызовите setup() перед train_dataloader().")
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """
        Создание загрузчика валидационных данных
        """
        if self.val_dataset is None:
            raise RuntimeError("Датасет не инициализирован. Вызовите setup() перед val_dataloader().")
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def test_dataloader(self) -> DataLoader:
        """
        Создание загрузчика тестовых данных
        """
        if self.test_dataset is None:
            raise RuntimeError("Датасет не инициализирован. Вызовите setup() перед test_dataloader().")
        
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )