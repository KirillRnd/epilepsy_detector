import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from src.data_loading.augmentations import EEGAugmentor    

class EpilepsyDataset_v2(Dataset):
    """
    Датасет для загрузки предобработанных данных ЭЭГ
    """

    def __init__(self, data_dir: str, segments_df: pd.DataFrame,
                 window_length: int = 2000, overlap: float = 0.5, augmentor: EEGAugmentor = None):
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
        
        self.augmentor = augmentor

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
        
        # Применяем аугментации (только для train!)
        if self.augmentor is not None:
            window_tensor, target_tensor = self.augmentor(window_tensor, target_tensor)
        
        return window_tensor, target_tensor