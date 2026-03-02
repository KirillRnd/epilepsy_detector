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
        self.log_flag = True
        
                # Создаем кэш для хранения данных в памяти
        self.data_cache = {}
        
        # Предзагрузка всех уникальных файлов
        print("Предзагрузка данных в оперативную память...")
        for animal_id, session_id, _, _, _ in self.windows:
            cache_key = (animal_id, session_id)
            if cache_key not in self.data_cache:
                data_file = self.data_dir / animal_id / session_id / "processed_signals.npy"
                # Загружаем и сохраняем в кэш
                self.data_cache[cache_key] = np.load(data_file)
        print("Предзагрузка завершена!")
    
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
        
        # Получаем данные из кэша в оперативной памяти
        data = self.data_cache[(animal_id, session_id)]
        
        # Извлечение окна
        window_data = data[:, start_sample:end_sample]
        
        # Преобразование в тензор
        window_tensor = torch.FloatTensor(window_data)
        
        # Вычисляем текущую длину и необходимый паддинг
        current_length = window_tensor.shape[1]
        
        if current_length < self.window_length:
            pad_size = self.window_length - current_length
            # F.pad принимает аргумент в виде (pad_left, pad_right) для последней размерности
            # Добавляем нули только в конец (справа)
            window_tensor = torch.nn.functional.pad(window_tensor, (0, pad_size), mode='constant', value=0.0)
            
        elif current_length > self.window_length:
            # На всякий случай, если окно вдруг оказалось больше (обрезаем лишнее)
            window_tensor = window_tensor[:, :self.window_length]
            
        if self.log_flag:
            print(f"Loaded window with shape {window_tensor.shape}")
            self.log_flag = False
        
        return window_tensor, label
    
class EpilepsyDataset_v2(Dataset):
    """
    Датасет для загрузки предобработанных данных ЭЭГ
    """

    def __init__(self, data_dir: str, segments_df: pd.DataFrame,
                 window_length: int = 2000, overlap: float = 0.5):
        self.data_dir = Path(data_dir)
        self.segments_df = segments_df
        self.window_length = window_length
        self.overlap = overlap
        self.step_size = int(window_length * (1 - overlap))

        # Сохраняем seizure-интервалы по сессиям (для быстрого построения target в __getitem__)
        self.seizure_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for (animal_id, session_id), g in self.segments_df.groupby(['animal_id', 'session_id']):
            seiz = g[g['segment_type'] == 'seizure'][['start_sample', 'end_sample']].values
            self.seizure_intervals[(animal_id, session_id)] = [(int(s), int(e)) for s, e in seiz]

        # Создание списка окон по всей записи (с учётом краёв)
        self.windows = self._create_windows()

        # Кэш сигналов в памяти
        self.data_cache = {}

        print("Предзагрузка данных в оперативную память...")
        for animal_id, session_id, _, _ in self.windows:
            cache_key = (animal_id, session_id)
            if cache_key not in self.data_cache:
                data_file = self.data_dir / animal_id / session_id / "processed_signals.npy"
                self.data_cache[cache_key] = np.load(data_file)
        print("Предзагрузка завершена!")

    def _create_windows(self) -> List[Tuple[str, str, int, int]]:
        """
        Создание списка окон по всей записи, включая края.

        Возвращает:
        list: список кортежей (animal_id, session_id, start_sample, end_sample),
              где end_sample может быть < start_sample + window_length (последнее окно у края).
        """
        windows: List[Tuple[str, str, int, int]] = []

        # Берём уникальные сессии из таблицы сегментов
        sessions = self.segments_df[['animal_id', 'session_id']].drop_duplicates().itertuples(index=False)

        for s in sessions:
            animal_id = s.animal_id
            session_id = s.session_id

            data_file = self.data_dir / animal_id / session_id / "processed_signals.npy"
            # mmap, чтобы только узнать длину записи без загрузки всего массива
            arr = np.load(data_file, mmap_mode='r')
            n_samples = int(arr.shape[1])

            if n_samples <= 0:
                continue

            # Стартовые позиции с перекрытием + обязательное окно, которое "дотягивает" до конца
            max_start = max(0, n_samples - self.window_length)

            if self.step_size <= 0:
                raise ValueError("step_size <= 0. Проверьте window_length и overlap.")

            starts = list(range(0, max_start + 1, self.step_size))
            if len(starts) == 0:
                starts = [0]
            if starts[-1] != max_start:
                starts.append(max_start)

            for start in starts:
                end = min(start + self.window_length, n_samples)
                windows.append((animal_id, session_id, int(start), int(end)))

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int):
        """
        Возвращает:
        (signal_window, target_window)
        signal_window: Tensor (C, window_length)
        target_window: Tensor (window_length,) из 0/1
        """
        animal_id, session_id, start_sample, end_sample = self.windows[idx]

        data = self.data_cache[(animal_id, session_id)]
        window_data = data[:, start_sample:end_sample]  # (C, L<=window_length)
        cur_len = window_data.shape[1]

        # pad справа, если это край и окно короче window_length
        if cur_len < self.window_length:
            pad = self.window_length - cur_len
            window_data = np.pad(window_data, ((0, 0), (0, pad)), mode='constant', constant_values=0.0)

        # Формируем target длиной window_length по seizure-интервалам
        target = np.zeros((self.window_length,), dtype=np.float32)

        for s0, s1 in self.seizure_intervals.get((animal_id, session_id), []):
            # пересечение [start_sample, end_sample) с [s0, s1)
            ov_start = max(start_sample, s0)
            ov_end = min(end_sample, s1)
            if ov_end > ov_start:
                a = ov_start - start_sample
                b = ov_end - start_sample
                target[a:b] = 1.0

        window_tensor = torch.from_numpy(window_data).float()
        target_tensor = torch.from_numpy(target).float()
        return window_tensor, target_tensor