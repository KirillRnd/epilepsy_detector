import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Sampler
import pandas as pd
from pathlib import Path
from typing import Iterator, Optional, List
from sklearn.model_selection import train_test_split
import numpy as np
import random
from src.data_loading.augmentations import EEGAugmentor, MixupCutMixCollator
from .epilepsy_dataset import EpilepsyDataset_v2,EpilepsyDataset_v3 


class WindowBlockShuffleSampler(Sampler[int]):
    def __init__(self, data_source, block_size: int = 4096, seed: int = 42):
        if block_size < 1:
            raise ValueError("block_size must be >= 1")

        self.data_source = data_source
        self.block_size = int(block_size)
        self.seed = int(seed)
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        n_items = len(self.data_source)
        blocks = [
            (start, min(start + self.block_size, n_items))
            for start in range(0, n_items, self.block_size)
        ]
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(blocks)
        self.epoch += 1

        for start, end in blocks:
            yield from range(start, end)

    def __len__(self) -> int:
        return len(self.data_source)


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
                 seed: int = 42,
                 cache_mode: str = "mmap",
                 max_open_files: int = 16,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 prefetch_factor: int = 2,
                 train_shuffle_mode: str = "block",
                 block_shuffle_size: int = 4096,
                 input_normalization: str = "none",
                 normalization_stats_path: str | None = None):
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
        self.cache_mode = cache_mode
        self.max_open_files = max_open_files
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor
        self.train_shuffle_mode = train_shuffle_mode
        self.block_shuffle_size = block_shuffle_size
        self.input_normalization = str(input_normalization or "none")
        self.normalization_stats_path = normalization_stats_path
        
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
        
        augmentor = EEGAugmentor(
            p_noise=0.5, noise_std_range=(0.01, 0.1),
            p_scale=0.5, scale_range=(0.7, 1.3),
            p_time_shift=0.3, max_shift_samples=80,
            p_channel_dropout=0.2,
            label_smooth_samples=40,
        )
        
        # Создание датасетов
        if stage == "fit" or stage is None:
            self.train_dataset = EpilepsyDataset_v3(
                data_dir=str(self.data_dir),
                segments_df=train_segments,
                window_length=self.window_length,
                overlap=self.overlap,               
                seizure_overlap=0.9,   # <-- добавить
                augmentor=augmentor,
                cache_mode=self.cache_mode,
                max_open_files=self.max_open_files,
                input_normalization=self.input_normalization,
                normalization_stats_path=self.normalization_stats_path,
            )
            
            self.val_dataset = EpilepsyDataset_v2(
                data_dir=str(self.data_dir),
                segments_df=val_segments,
                window_length=self.window_length,
                overlap=self.overlap,
                augmentor=None,  # <-- без аугментации
                cache_mode=self.cache_mode,
                max_open_files=self.max_open_files,
                input_normalization=self.input_normalization,
                normalization_stats_path=self.normalization_stats_path,
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = EpilepsyDataset_v2(
                data_dir=str(self.data_dir),
                segments_df=test_segments,
                window_length=self.window_length,
                overlap=self.overlap,
                augmentor=None,  # <-- без аугментации
                cache_mode=self.cache_mode,
                max_open_files=self.max_open_files,
                input_normalization=self.input_normalization,
                normalization_stats_path=self.normalization_stats_path,
            )
        
        # Вывод разбиения животных на сеты
        print(f"Train animals: {train_animals}")
        print(f"Validation animals: {val_animals}")
        print(f"Test animals: {test_animals}")
        print(f"Input normalization: {self.input_normalization}")

    def _dataloader_kwargs(self):
        kwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = self.persistent_workers
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs
    
    def train_dataloader(self) -> DataLoader:
        """
        Создание загрузчика обучающих данных
        """
        if self.train_dataset is None:
            raise RuntimeError("Датасет не инициализирован. Вызовите setup() перед train_dataloader().")
        
        collator = MixupCutMixCollator(
            p_mixup=0.3, mixup_alpha=0.3,
            p_cutmix=0.2, cutmix_min_len=200, cutmix_max_len=800,
        )

        if self.train_shuffle_mode == "block":
            sampler = WindowBlockShuffleSampler(
                self.train_dataset,
                block_size=self.block_shuffle_size,
                seed=self.seed,
            )
            shuffle = False
        elif self.train_shuffle_mode == "full":
            sampler = None
            shuffle = True
        elif self.train_shuffle_mode == "none":
            sampler = None
            shuffle = False
        else:
            raise ValueError(
                "train_shuffle_mode must be one of: 'block', 'full', 'none'"
            )
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            collate_fn=collator,  # <-- Mixup/CutMix на уровне батча
            **self._dataloader_kwargs(),
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
            **self._dataloader_kwargs(),
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
            **self._dataloader_kwargs(),
        )
