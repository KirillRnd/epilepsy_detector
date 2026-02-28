import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import os

class EpilepsyDataset(Dataset):
    """
    Датасет для загрузки предобработанных данных ЭЭГ
    """
    
    def __init__(self, data_dir: str, segments_df: pd.DataFrame, 
                 window_length: int = 2000, overlap: float = 0.5):
        """
        Инициализация датасета
        
        Параметры:
        data_dir (str): директория с предобработанными данными
        segments_df (pd.DataFrame): DataFrame с информацией о сегментах
        window_length (int): длина окна в отсчетах (по умолчанию 2000 отсчетов = 5 сек при 400 Гц)
        overlap (float): перекрытие окон (0.0 - 1.0)
        """
        self.data_dir = Path(data_dir)
        self.segments_df = segments_df
        self.window_length = window_length
        self.overlap = overlap
        self.step_size = int(window_length * (1 - overlap))
        
        # Создание списка окон
        self.windows = self._create_windows()
    
    def _create_windows(self) -> List[Tuple[str, str, int, int, int]]:
        """
        Создание списка окон для загрузки
        
        Возвращает:
        list: список кортежей (animal_id, session_id, start_sample, end_sample, label)
        """
        windows = []
        
        for _, row in self.segments_df.iterrows():
            animal_id = row['animal_id']
            session_id = row['session_id']
            segment_type = row['segment_type']
            start_sample = int(row['start_sample'])
            end_sample = int(row['end_sample'])
            
            # Определение метки класса
            label = 1 if segment_type == 'seizure' else 0
            
            # Создание окон в сегменте
            current_start = start_sample
            while current_start + self.window_length <= end_sample:
                current_end = current_start + self.window_length
                windows.append((animal_id, session_id, current_start, current_end, label))
                current_start += self.step_size
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Получение элемента датасета
        
        Параметры:
        idx (int): индекс элемента
        
        Возвращает:
        tuple: (сигнал, метка)
        """
        animal_id, session_id, start_sample, end_sample, label = self.windows[idx]
        
        # Путь к файлу с данными
        data_file = self.data_dir / animal_id / session_id / "processed_signals.npy"
        
        # Загрузка данных
        data = np.load(data_file)
        
        # Извлечение окна
        window_data = data[:, start_sample:end_sample]
        
        # Преобразование в тензор
        window_tensor = torch.FloatTensor(window_data)
        
        return window_tensor, label