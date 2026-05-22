import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import csv
from pathlib import Path
from collections import OrderedDict
from typing import List, Tuple, Dict
from src.data_loading.augmentations import EEGAugmentor    
from src.data_loading.input_normalization import (
    apply_precomputed_input_normalization,
    fit_input_normalization,
)


class LazySignalMixin:
    def _init_signal_loading(
        self,
        cache_mode: str = "mmap",
        max_open_files: int = 16,
        input_normalization: str = "none",
        normalization_stats_path: str | None = None,
    ):
        if cache_mode not in {"mmap", "eager"}:
            raise ValueError("cache_mode must be 'mmap' or 'eager'")
        if max_open_files < 1:
            raise ValueError("max_open_files must be >= 1")

        self.cache_mode = cache_mode
        self.max_open_files = int(max_open_files)
        self.input_normalization = str(input_normalization or "none")
        self.normalization_stats_path = (
            Path(normalization_stats_path) if normalization_stats_path else None
        )
        self.data_cache = OrderedDict()
        self.normalization_params = {}
        self._normalization_stats_written = set()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["data_cache"] = OrderedDict()
        return state

    def _data_file(self, animal_id: str, session_id: str) -> Path:
        return self.data_dir / animal_id / session_id / "processed_signals.npy"

    def _preload_signals_if_needed(self):
        if self.cache_mode != "eager":
            print(
                f"Lazy signal loading enabled: cache_mode={self.cache_mode}, "
                f"max_open_files={self.max_open_files}, "
                f"input_normalization={self.input_normalization}"
            )
            return

        print(f"Preloading signals into RAM with input_normalization={self.input_normalization}...")
        seen = set()
        for animal_id, session_id, _, _ in self.windows:
            cache_key = (animal_id, session_id)
            if cache_key in seen:
                continue
            seen.add(cache_key)
            self.data_cache[cache_key] = np.load(self._data_file(animal_id, session_id))
        print("Preloading complete!")

    def _write_normalization_stats(
        self,
        animal_id: str,
        session_id: str,
        stats: dict,
    ) -> None:
        if self.normalization_stats_path is None:
            return
        cache_key = (animal_id, session_id)
        if cache_key in self._normalization_stats_written:
            return

        self.normalization_stats_path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.normalization_stats_path.exists()
        fieldnames = [
            "animal_id",
            "session_id",
            "recording_id",
            "input_normalization",
            "channel",
            "center",
            "scale",
            "raw_mean",
            "raw_std",
            "norm_mean",
            "norm_std",
            "warning_flags",
        ]
        existing_keys = set()
        if file_exists:
            with self.normalization_stats_path.open("r", encoding="utf-8", newline="") as handle:
                for row in csv.DictReader(handle):
                    existing_keys.add(
                        (
                            row.get("animal_id"),
                            row.get("session_id"),
                            row.get("input_normalization"),
                            row.get("channel"),
                        )
                    )

        with self.normalization_stats_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in stats.get("rows", []):
                row_key = (
                    animal_id,
                    session_id,
                    row.get("input_normalization"),
                    row.get("channel"),
                )
                if row_key in existing_keys:
                    continue
                writer.writerow(
                    {
                        "animal_id": animal_id,
                        "session_id": session_id,
                        "recording_id": f"{animal_id}_{session_id}",
                        **row,
                    }
                )
        self._normalization_stats_written.add(cache_key)

    def _unique_recordings(self):
        seen = set()
        for animal_id, session_id, _, _ in self.windows:
            cache_key = (animal_id, session_id)
            if cache_key in seen:
                continue
            seen.add(cache_key)
            yield animal_id, session_id

    def _precompute_normalization_if_needed(self) -> None:
        print(f"Precomputing input normalization stats: {self.input_normalization}")
        for animal_id, session_id in self._unique_recordings():
            cache_key = (animal_id, session_id)
            if cache_key in self.normalization_params:
                continue
            if self.input_normalization == "none" and self.normalization_stats_path is None:
                self.normalization_params[cache_key] = {
                    "input_normalization": "none",
                    "center": np.zeros((1, 1), dtype=np.float32),
                    "scale": np.ones((1, 1), dtype=np.float32),
                    "warnings": [""],
                }
                continue
            data = np.load(self._data_file(animal_id, session_id), mmap_mode="r")
            params, stats = fit_input_normalization(
                data,
                self.input_normalization,
                return_stats=True,
            )
            self.normalization_params[cache_key] = params
            self._write_normalization_stats(animal_id, session_id, stats)

    def _normalize_window(self, window_data, animal_id: str, session_id: str):
        params = self.normalization_params.get((animal_id, session_id))
        if params is None:
            raise RuntimeError(
                "Input normalization parameters were not precomputed for "
                f"{animal_id}/{session_id}"
            )
        return apply_precomputed_input_normalization(window_data, params)

    def _get_signal_array(self, animal_id: str, session_id: str):
        cache_key = (animal_id, session_id)
        data = self.data_cache.get(cache_key)
        if data is not None:
            self.data_cache.move_to_end(cache_key)
            return data

        mmap_mode = "r" if self.cache_mode == "mmap" else None
        data = np.load(self._data_file(animal_id, session_id), mmap_mode=mmap_mode)
        self.data_cache[cache_key] = data
        self.data_cache.move_to_end(cache_key)

        if self.cache_mode == "mmap":
            while len(self.data_cache) > self.max_open_files:
                self.data_cache.popitem(last=False)

        return data

class EpilepsyDataset_v2(LazySignalMixin, Dataset):
    """
    Датасет для загрузки предобработанных данных ЭЭГ
    """

    def __init__(self, data_dir: str, segments_df: pd.DataFrame,
                 window_length: int = 2000, overlap: float = 0.5,
                 augmentor: EEGAugmentor = None,
                 cache_mode: str = "mmap", max_open_files: int = 16,
                 input_normalization: str = "none",
                 normalization_stats_path: str | None = None):
        self.data_dir = Path(data_dir)
        self.segments_df = segments_df
        self.window_length = window_length
        self.overlap = overlap
        self.step_size = int(window_length * (1 - overlap))
        self._init_signal_loading(
            cache_mode=cache_mode,
            max_open_files=max_open_files,
            input_normalization=input_normalization,
            normalization_stats_path=normalization_stats_path,
        )

        # Сохраняем seizure-интервалы по сессиям (для быстрого построения target в __getitem__)
        self.seizure_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for (animal_id, session_id), g in self.segments_df.groupby(['animal_id', 'session_id']):
            seiz = g[g['segment_type'] == 'seizure'][['start_sample', 'end_sample']].values
            self.seizure_intervals[(animal_id, session_id)] = [(int(s), int(e)) for s, e in seiz]

        # Создание списка окон по всей записи (с учётом краёв)
        self.windows = self._create_windows()

        self._precompute_normalization_if_needed()
        self._preload_signals_if_needed()
        
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

        windows.sort(key=lambda w: (w[0], w[1], w[2], w[3]))
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

        data = self._get_signal_array(animal_id, session_id)
        window_data = data[:, start_sample:end_sample]  # (C, L<=window_length)
        cur_len = window_data.shape[1]
        window_data = self._normalize_window(window_data, animal_id, session_id)

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
    
class EpilepsyDataset_v3(LazySignalMixin, Dataset):

    def __init__(
        self,
        data_dir: str,
        segments_df: pd.DataFrame,
        window_length: int = 2000,
        overlap: float = 0.5,
        seizure_overlap: float = 0.9,   # <-- новый параметр
        augmentor: EEGAugmentor = None,
        cache_mode: str = "mmap",
        max_open_files: int = 16,
        input_normalization: str = "none",
        normalization_stats_path: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.segments_df = segments_df
        self.window_length = window_length
        self.overlap = overlap
        self.step_size = int(window_length * (1 - overlap))
        self.seizure_step_size = int(window_length * (1 - seizure_overlap))
        self._init_signal_loading(
            cache_mode=cache_mode,
            max_open_files=max_open_files,
            input_normalization=input_normalization,
            normalization_stats_path=normalization_stats_path,
        )

        # Seizure-интервалы по сессиям (для target и для плотного семплирования)
        self.seizure_intervals: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
        for (animal_id, session_id), g in self.segments_df.groupby(['animal_id', 'session_id']):
            seiz = g[g['segment_type'] == 'seizure'][['start_sample', 'end_sample']].values
            self.seizure_intervals[(animal_id, session_id)] = [
                (int(s), int(e)) for s, e in seiz
            ]

        self.windows = self._create_windows()

        self._precompute_normalization_if_needed()
        self._preload_signals_if_needed()

        self.augmentor = augmentor

    def _create_windows(self) -> List[Tuple[str, str, int, int]]:
        windows: List[Tuple[str, str, int, int]] = []

        sessions = (
            self.segments_df[['animal_id', 'session_id']]
            .drop_duplicates()
            .itertuples(index=False)
        )

        for sess in sessions:
            animal_id  = sess.animal_id
            session_id = sess.session_id

            data_file = self.data_dir / animal_id / session_id / "processed_signals.npy"
            arr = np.load(data_file, mmap_mode='r')
            n_samples = int(arr.shape[1])
            if n_samples <= 0:
                continue

            seizure_ivs = self.seizure_intervals.get((animal_id, session_id), [])
            max_start = max(0, n_samples - self.window_length)

            # Множество уже добавленных позиций — защита от дублей
            added = set()  # без аннотации типа

            def add_window(start):
                start = int(max(0, min(start, max_start)))
                if start in added:
                    return
                added.add(start)
                end = min(start + self.window_length, n_samples)
                windows.append((animal_id, session_id, start, end))

            # 1. Обычный шаг по всей записи
            pos = 0
            while pos <= max_start:
                add_window(pos)
                pos += self.step_size
            # Гарантируем последнее окно у края
            add_window(max_start)

            # 2. Плотный шаг внутри и вокруг каждого приступа
            for s0, s1 in seizure_ivs:
                dense_start = max(0,          s0 - self.window_length)
                dense_end   = min(max_start,  s1 + self.window_length)

                pos = dense_start
                while pos <= dense_end:
                    add_window(pos)
                    pos += self.seizure_step_size
                # Гарантируем последнюю позицию зоны
                add_window(dense_end)

        windows.sort(key=lambda w: (w[0], w[1], w[2], w[3]))
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

        data = self._get_signal_array(animal_id, session_id)
        window_data = data[:, start_sample:end_sample]  # (C, L<=window_length)
        cur_len = window_data.shape[1]
        window_data = self._normalize_window(window_data, animal_id, session_id)

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
