import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path
from typing import Optional, List
from sklearn.model_selection import train_test_split
import numpy as np

from .epilepsy_dataset import EpilepsyDataset

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
        np.random.seed(self.seed)
        np.random.shuffle(unique_animals)
        
        n_train = int(len(unique_animals) * self.train_animal_ratio)
        n_val = int(len(unique_animals) * self.val_animal_ratio)
        
        train_animals = unique_animals[:n_train]
        val_animals = unique_animals[n_train:n_train+n_val]
        test_animals = unique_animals[n_train+n_val:]
        
        # Фильтрация сегментов по животным
        train_segments = all_segments_df[all_segments_df['animal_id'].isin(train_animals)]
        val_segments = all_segments_df[all_segments_df['animal_id'].isin(val_animals)]
        test_segments = all_segments_df[all_segments_df['animal_id'].isin(test_animals)]
        
        # Создание датасетов
        if stage == "fit" or stage is None:
            self.train_dataset = EpilepsyDataset(
                data_dir=str(self.data_dir),
                segments_df=train_segments,
                window_length=self.window_length,
                overlap=self.overlap
            )
            
            self.val_dataset = EpilepsyDataset(
                data_dir=str(self.data_dir),
                segments_df=val_segments,
                window_length=self.window_length,
                overlap=self.overlap
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = EpilepsyDataset(
                data_dir=str(self.data_dir),
                segments_df=test_segments,
                window_length=self.window_length,
                overlap=self.overlap
            )
    
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