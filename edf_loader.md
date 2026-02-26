# Загрузчик данных для .edf файлов

## Общая архитектура загрузчика

Загрузчик данных для .edf файлов должен обеспечивать:
1. Чтение сырых данных из .edf файлов
2. Предоставление унифицированного интерфейса для доступа к данным
3. Поддержку различных параметров предобработки
4. Эффективное использование памяти

## Базовая реализация загрузчика

```python
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

class EDFLoader:
    """
    Загрузчик данных для .edf файлов
    """
    
    def __init__(self, preload_data: bool = False):
        """
        Инициализация загрузчика
        
        Параметры:
        preload_data (bool): загружать данные в память сразу
        """
        self.preload_data = preload_data
        self.loaded_files = {}
    
    def load_file(self, file_path: str) -> Dict:
        """
        Загрузка .edf файла
        
        Параметры:
        file_path (str): путь к .edf файлу
        
        Возвращает:
        dict: словарь с данными и метаданными
        """
        # Проверка существования файла
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл не найден: {file_path}")
        
        # Загрузка файла с помощью mne
        try:
            raw = mne.io.read_raw_edf(file_path, preload=self.preload_data)
            
            # Извлечение метаданных
            metadata = {
                'file_path': file_path,
                'sampling_freq': raw.info['sfreq'],
                'n_channels': len(raw.ch_names),
                'channel_names': raw.ch_names,
                'duration': raw.times[-1] if len(raw.times) > 0 else 0,
                'raw_object': raw
            }
            
            # Сохранение ссылки на загруженный файл
            if self.preload_data:
                self.loaded_files[file_path] = {
                    'raw': raw,
                    'data': raw.get_data(),
                    'metadata': metadata
                }
            
            return metadata
            
        except Exception as e:
            raise Exception(f"Ошибка при загрузке файла {file_path}: {str(e)}")
    
    def get_data(self, file_path: str, channels: Optional[List[str]] = None) -> np.ndarray:
        """
        Получение данных из загруженного файла
        
        Параметры:
        file_path (str): путь к файлу
        channels (list): список каналов для извлечения (если None, то все каналы)
        
        Возвращает:
        np.ndarray: массив с данными
        """
        if self.preload_data and file_path in self.loaded_files:
            raw = self.loaded_files[file_path]['raw']
        else:
            # Загрузка файла при необходимости
            if file_path not in self.loaded_files:
                self.load_file(file_path)
            raw = self.loaded_files[file_path]['raw']
        
        # Извлечение данных
        if channels is None:
            return raw.get_data()
        else:
            # Выбор определенных каналов
            picks = [raw.ch_names.index(ch) for ch in channels if ch in raw.ch_names]
            return raw.get_data(picks=picks)
    
    def get_channel_names(self, file_path: str) -> List[str]:
        """
        Получение имен каналов из файла
        
        Параметры:
        file_path (str): путь к файлу
        
        Возвращает:
        list: список имен каналов
        """
        if file_path in self.loaded_files:
            return self.loaded_files[file_path]['metadata']['channel_names']
        else:
            metadata = self.load_file(file_path)
            return metadata['channel_names']
    
    def get_sampling_frequency(self, file_path: str) -> float:
        """
        Получение частоты дискретизации из файла
        
        Параметры:
        file_path (str): путь к файлу
        
        Возвращает:
        float: частота дискретизации
        """
        if file_path in self.loaded_files:
            return self.loaded_files[file_path]['metadata']['sampling_freq']
        else:
            metadata = self.load_file(file_path)
            return metadata['sampling_freq']
    
    def close_file(self, file_path: str) -> None:
        """
        Освобождение ресурсов, связанных с файлом
        
        Параметры:
        file_path (str): путь к файлу
        """
        if file_path in self.loaded_files:
            del self.loaded_files[file_path]
    
    def close_all(self) -> None:
        """
        Освобождение всех ресурсов
        """
        self.loaded_files.clear()
```

## Расширенная реализация с поддержкой предобработки

```python
class AdvancedEDFLoader(EDFLoader):
    """
    Расширенный загрузчик данных для .edf файлов с поддержкой предобработки
    """
    
    def __init__(self, preload_data: bool = False, 
                 target_sampling_rate: Optional[float] = None,
                 channel_selection: Optional[List[str]] = None):
        """
        Инициализация расширенного загрузчика
        
        Параметры:
        preload_data (bool): загружать данные в память сразу
        target_sampling_rate (float): целевая частота дискретизации (для ресемплинга)
        channel_selection (list): список каналов для выбора по умолчанию
        """
        super().__init__(preload_data)
        self.target_sampling_rate = target_sampling_rate
        self.channel_selection = channel_selection
    
    def load_and_preprocess(self, file_path: str, 
                          apply_filter: bool = True,
                          low_freq: float = 1.0,
                          high_freq: float = 100.0,
                          apply_notch: bool = True,
                          notch_freq: float = 50.0) -> Dict:
        """
        Загрузка и предварительная обработка .edf файла
        
        Параметры:
        file_path (str): путь к .edf файлу
        apply_filter (bool): применять полосовой фильтр
        low_freq (float): нижняя частота полосового фильтра
        high_freq (float): верхняя частота полосового фильтра
        apply_notch (bool): применять режекторный фильтр
        notch_freq (float): частота режекторного фильтра
        
        Возвращает:
        dict: словарь с обработанными данными и метаданными
        """
        # Загрузка файла
        metadata = self.load_file(file_path)
        raw = metadata['raw_object']
        
        # Применение предобработки
        processed_raw = raw.copy()
        
        # Применение полосового фильтра
        if apply_filter:
            processed_raw.filter(l_freq=low_freq, h_freq=high_freq, 
                               method='iir', iir_params={'order': 4, 'ftype': 'butter'})
        
        # Применение режекторного фильтра
        if apply_notch:
            processed_raw.notch_filter(freqs=notch_freq)
        
        # Ресемплинг при необходимости
        if self.target_sampling_rate and self.target_sampling_rate != metadata['sampling_freq']:
            processed_raw.resample(self.target_sampling_rate)
            metadata['sampling_freq'] = self.target_sampling_rate
        
        # Выбор каналов при необходимости
        if self.channel_selection:
            # Проверка наличия каналов
            available_channels = [ch for ch in self.channel_selection if ch in processed_raw.ch_names]
            if available_channels:
                processed_raw.pick_channels(available_channels)
                metadata['channel_names'] = available_channels
                metadata['n_channels'] = len(available_channels)
        
        # Обновление метаданных
        metadata['processed_raw'] = processed_raw
        metadata['processed_data'] = processed_raw.get_data()
        
        return metadata
    
    def extract_windows(self, file_path: str, 
                       window_length: float = 5.0,
                       window_overlap: float = 0.0,
                       preprocess: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Извлечение временных окон из файла
        
        Параметры:
        file_path (str): путь к файлу
        window_length (float): длина окна в секундах
        window_overlap (float): перекрытие окон (0.0 - 1.0)
        preprocess (bool): применять предобработку
        
        Возвращает:
        tuple: (массив окон, список метаданных окон)
        """
        # Загрузка и предобработка данных
        if preprocess:
            metadata = self.load_and_preprocess(file_path)
            raw = metadata['processed_raw']
        else:
            metadata = self.load_file(file_path)
            raw = metadata['raw_object']
        
        # Параметры окон
        sfreq = raw.info['sfreq']
        window_samples = int(window_length * sfreq)
        step_samples = int(window_samples * (1.0 - window_overlap))
        
        # Извлечение данных
        data = raw.get_data()
        
        # Создание окон
        windows = []
        window_metadata = []
        
        start_sample = 0
        window_index = 0
        
        while start_sample + window_samples <= data.shape[1]:
            # Извлечение окна
            end_sample = start_sample + window_samples
            window_data = data[:, start_sample:end_sample]
            windows.append(window_data)
            
            # Создание метаданных окна
            start_time = start_sample / sfreq
            end_time = end_sample / sfreq
            
            window_meta = {
                'window_index': window_index,
                'file_path': file_path,
                'start_time': start_time,
                'end_time': end_time,
                'duration': window_length,
                'start_sample': start_sample,
                'end_sample': end_sample
            }
            
            window_metadata.append(window_meta)
            
            # Переход к следующему окну
            start_sample += step_samples
            window_index += 1
        
        return np.array(windows), window_metadata
```

## Использование загрузчика

```python
# Пример использования базового загрузчика
def example_basic_usage():
    """
    Пример использования базового загрузчика
    """
    # Создание загрузчика
    loader = EDFLoader(preload_data=False)
    
    # Загрузка файла
    file_path = "data/raw/Dex1y2/BL_10May/Dex1y2_10May_BL.edf"
    metadata = loader.load_file(file_path)
    
    print(f"Файл: {metadata['file_path']}")
    print(f"Частота дискретизации: {metadata['sampling_freq']} Гц")
    print(f"Количество каналов: {metadata['n_channels']}")
    print(f"Длительность: {metadata['duration']} сек")
    print(f"Каналы: {metadata['channel_names']}")
    
    # Получение данных
    data = loader.get_data(file_path)
    print(f"Размер данных: {data.shape}")
    
    # Освобождение ресурсов
    loader.close_file(file_path)

# Пример использования расширенного загрузчика
def example_advanced_usage():
    """
    Пример использования расширенного загрузчика
    """
    # Создание расширенного загрузчика
    loader = AdvancedEDFLoader(
        preload_data=True,
        target_sampling_rate=512.0,
        channel_selection=['EEG1', 'EEG2']
    )
    
    # Загрузка и предобработка файла
    file_path = "data/raw/Dex1y2/BL_10May/Dex1y2_10May_BL.edf"
    metadata = loader.load_and_preprocess(
        file_path,
        apply_filter=True,
        low_freq=1.0,
        high_freq=100.0,
        apply_notch=True,
        notch_freq=50.0
    )
    
    print(f"Обработанные данные: {metadata['processed_data'].shape}")
    
    # Извлечение временных окон
    windows, window_meta = loader.extract_windows(
        file_path,
        window_length=5.0,
        window_overlap=0.5,
        preprocess=False  # Уже предобработано
    )
    
    print(f"Количество окон: {len(windows)}")
    print(f"Размер окна: {windows[0].shape}")
    print(f"Метаданные первого окна: {window_meta[0]}")

# Пример пакетной обработки
def batch_processing_example(file_list: List[str]):
    """
    Пример пакетной обработки файлов
    
    Параметры:
    file_list (list): список путей к файлам
    """
    loader = AdvancedEDFLoader(preload_data=False)
    
    results = []
    
    for file_path in file_list:
        try:
            # Загрузка файла
            metadata = loader.load_file(file_path)
            
            # Сохранение результатов
            results.append({
                'file_path': file_path,
                'success': True,
                'sampling_freq': metadata['sampling_freq'],
                'n_channels': metadata['n_channels'],
                'duration': metadata['duration']
            })
            
        except Exception as e:
            # Сохранение информации об ошибке
            results.append({
                'file_path': file_path,
                'success': False,
                'error': str(e)
            })
        
        finally:
            # Освобождение ресурсов
            loader.close_file(file_path)
    
    return results
```

## Интеграция с системой метаданных

```python
class IntegratedEDFLoader(AdvancedEDFLoader):
    """
    Загрузчик с интеграцией системы метаданных
    """
    
    def __init__(self, metadata_file: str, **kwargs):
        """
        Инициализация загрузчика с метаданными
        
        Параметры:
        metadata_file (str): путь к файлу метаданных
        **kwargs: дополнительные параметры для родительского класса
        """
        super().__init__(**kwargs)
        self.metadata_df = pd.read_csv(metadata_file)
    
    def get_file_metadata(self, file_path: str) -> Dict:
        """
        Получение метаданных файла из таблицы
        
        Параметры:
        file_path (str): путь к файлу
        
        Возвращает:
        dict: метаданные файла
        """
        # Поиск в таблице метаданных
        row = self.metadata_df[self.metadata_df['file_path'] == file_path]
        
        if len(row) > 0:
            return row.iloc[0].to_dict()
        else:
            return {}
    
    def load_with_metadata(self, file_path: str) -> Dict:
        """
        Загрузка файла с интеграцией метаданных
        
        Параметры:
        file_path (str): путь к файлу
        
        Возвращает:
        dict: объединенные данные и метаданные
        """
        # Загрузка данных
        data_metadata = self.load_file(file_path)
        
        # Получение метаданных из таблицы
        file_metadata = self.get_file_metadata(file_path)
        
        # Объединение метаданных
        combined_metadata = {**data_metadata, **file_metadata}
        
        return combined_metadata
```

## Рекомендации по использованию

1. **Выбор режима загрузки**: Используйте `preload_data=True` для небольших файлов или при частом доступе к данным
2. **Управление памятью**: Регулярно вызывайте `close_file()` или `close_all()` для освобождения ресурсов
3. **Предобработка**: Используйте `AdvancedEDFLoader` для автоматической предобработки данных
4. **Интеграция метаданных**: Используйте `IntegratedEDFLoader` для работы с таблицами метаданных
5. **Пакетная обработка**: Используйте функции пакетной обработки для обработки множества файлов

## Возможные улучшения

1. **Кэширование**: Реализовать кэширование обработанных данных для ускорения повторного доступа
2. **Параллельная обработка**: Добавить поддержку многопоточной обработки файлов
3. **Мониторинг памяти**: Добавить функции мониторинга использования памяти
4. **Логирование**: Добавить подробное логирование операций загрузки и обработки