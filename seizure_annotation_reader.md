# Модуль для чтения разметки приступов из текстовых файлов

## Общая архитектура модуля

Модуль для чтения разметки приступов должен обеспечивать:
1. Чтение текстовых файлов с разметкой приступов
2. Парсинг формата разметки (пары значений: начало и конец приступа)
3. Предоставление унифицированного интерфейса для доступа к разметке
4. Конвертацию разметки в различные форматы (DataFrame, списки и т.д.)

## Базовая реализация модуля

```python
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

class SeizureAnnotationReader:
    """
    Модуль для чтения разметки приступов из текстовых файлов
    """
    
    def __init__(self):
        """
        Инициализация модуля
        """
        self.annotations = {}
    
    def load_annotation_file(self, file_path: str, 
                           file_id: Optional[str] = None) -> Dict:
        """
        Загрузка файла разметки приступов
        
        Параметры:
        file_path (str): путь к файлу разметки
        file_id (str): идентификатор файла (если не указан, используется имя файла)
        
        Возвращает:
        dict: словарь с разметкой приступов
        """
        # Проверка существования файла
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл разметки не найден: {file_path}")
        
        # Определение идентификатора файла
        if file_id is None:
            file_id = Path(file_path).stem
        
        # Чтение файла
        seizures = self._parse_annotation_file(file_path)
        
        # Сохранение разметки
        annotation_data = {
            'file_id': file_id,
            'file_path': file_path,
            'seizures': seizures,
            'count': len(seizures)
        }
        
        self.annotations[file_id] = annotation_data
        
        return annotation_data
    
    def _parse_annotation_file(self, file_path: str) -> List[Dict]:
        """
        Парсинг файла разметки приступов
        
        Параметры:
        file_path (str): путь к файлу разметки
        
        Возвращает:
        list: список словарей с информацией о приступах
        """
        seizures = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Парсинг строки с разметкой
                seizure_data = self._parse_seizure_line(line, line_num)
                if seizure_data:
                    seizures.append(seizure_data)
        
        except Exception as e:
            raise Exception(f"Ошибка при чтении файла {file_path}: {str(e)}")
        
        return seizures
    
    def _parse_seizure_line(self, line: str, line_num: int) -> Optional[Dict]:
        """
        Парсинг строки с информацией о приступе
        
        Параметры:
        line (str): строка с разметкой
        line_num (int): номер строки
        
        Возвращает:
        dict: словарь с информацией о приступе или None
        """
        # Поддерживаемые форматы:
        # 1. "start,end" (например, "55,1	57,1")
        # 2. "start end" (например, "55.1 57.1")
        # 3. "start,end,label" (например, "55.1,57.1,seizure")
        
        # Разделение по запятым или пробелам
        parts = re.split(r'[,;\s]+', line)
        parts = [p for p in parts if p]  # Удаление пустых элементов
        
        if len(parts) >= 2:
            try:
                start_time = float(parts[0].replace(',', '.'))
                end_time = float(parts[1].replace(',', '.'))
                
                # Проверка корректности временных меток
                if start_time >= 0 and end_time > start_time:
                    seizure_data = {
                        'start': start_time,
                        'end': end_time,
                        'duration': end_time - start_time,
                        'line_number': line_num
                    }
                    
                    # Добавление метки, если она есть
                    if len(parts) >= 3:
                        seizure_data['label'] = parts[2]
                    else:
                        seizure_data['label'] = 'seizure'
                    
                    return seizure_data
            except ValueError:
                # Игнорируем строки с некорректными числами
                pass
        
        return None
    
    def get_seizures(self, file_id: str) -> List[Dict]:
        """
        Получение списка приступов для файла
        
        Параметры:
        file_id (str): идентификатор файла
        
        Возвращает:
        list: список словарей с информацией о приступах
        """
        if file_id in self.annotations:
            return self.annotations[file_id]['seizures']
        else:
            return []
    
    def get_seizure_dataframe(self, file_id: str) -> pd.DataFrame:
        """
        Получение разметки приступов в виде DataFrame
        
        Параметры:
        file_id (str): идентификатор файла
        
        Возвращает:
        pd.DataFrame: DataFrame с разметкой приступов
        """
        seizures = self.get_seizures(file_id)
        
        if seizures:
            df = pd.DataFrame(seizures)
            df['file_id'] = file_id
            return df
        else:
            # Возвращаем пустой DataFrame с правильной структурой
            return pd.DataFrame(columns=['start', 'end', 'duration', 'label', 'line_number', 'file_id'])
    
    def get_all_seizures_dataframe(self) -> pd.DataFrame:
        """
        Получение разметки всех приступов в виде одного DataFrame
        
        Возвращает:
        pd.DataFrame: DataFrame со всей разметкой приступов
        """
        all_seizures = []
        
        for file_id, annotation_data in self.annotations.items():
            for seizure in annotation_data['seizures']:
                seizure_copy = seizure.copy()
                seizure_copy['file_id'] = file_id
                all_seizures.append(seizure_copy)
        
        if all_seizures:
            return pd.DataFrame(all_seizures)
        else:
            return pd.DataFrame(columns=['start', 'end', 'duration', 'label', 'line_number', 'file_id'])
    
    def get_seizure_count(self, file_id: str) -> int:
        """
        Получение количества приступов в файле
        
        Параметры:
        file_id (str): идентификатор файла
        
        Возвращает:
        int: количество приступов
        """
        if file_id in self.annotations:
            return self.annotations[file_id]['count']
        else:
            return 0
    
    def get_total_seizure_duration(self, file_id: str) -> float:
        """
        Получение общей длительности приступов в файле
        
        Параметры:
        file_id (str): идентификатор файла
        
        Возвращает:
        float: общая длительность приступов в секундах
        """
        seizures = self.get_seizures(file_id)
        return sum(seizure['duration'] for seizure in seizures)
    
    def clear_annotations(self, file_id: Optional[str] = None) -> None:
        """
        Очистка загруженной разметки
        
        Параметры:
        file_id (str): идентификатор файла (если None, очищает всю разметку)
        """
        if file_id is None:
            self.annotations.clear()
        elif file_id in self.annotations:
            del self.annotations[file_id]
```

## Расширенная реализация с поддержкой различных форматов

```python
class AdvancedSeizureAnnotationReader(SeizureAnnotationReader):
    """
    Расширенный модуль для чтения разметки приступов с поддержкой различных форматов
    """
    
    def __init__(self):
        """
        Инициализация расширенного модуля
        """
        super().__init__()
        self.supported_formats = {
            'simple': self._parse_simple_format,
            'bids': self._parse_bids_format,
            'eeglab': self._parse_eeglab_format,
            'custom': self._parse_custom_format
        }
    
    def load_annotation_file(self, file_path: str, 
                           file_id: Optional[str] = None,
                           format_type: str = 'simple') -> Dict:
        """
        Загрузка файла разметки приступов с указанием формата
        
        Параметры:
        file_path (str): путь к файлу разметки
        file_id (str): идентификатор файла
        format_type (str): тип формата ('simple', 'bids', 'eeglab', 'custom')
        
        Возвращает:
        dict: словарь с разметкой приступов
        """
        # Проверка существования файла
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Файл разметки не найден: {file_path}")
        
        # Определение идентификатора файла
        if file_id is None:
            file_id = Path(file_path).stem
        
        # Выбор функции парсинга в зависимости от формата
        if format_type in self.supported_formats:
            parse_func = self.supported_formats[format_type]
            seizures = parse_func(file_path)
        else:
            # По умолчанию используем простой формат
            seizures = self._parse_annotation_file(file_path)
        
        # Сохранение разметки
        annotation_data = {
            'file_id': file_id,
            'file_path': file_path,
            'format_type': format_type,
            'seizures': seizures,
            'count': len(seizures)
        }
        
        self.annotations[file_id] = annotation_data
        
        return annotation_data
    
    def _parse_simple_format(self, file_path: str) -> List[Dict]:
        """
        Парсинг простого формата разметки (наш стандартный формат)
        
        Параметры:
        file_path (str): путь к файлу разметки
        
        Возвращает:
        list: список словарей с информацией о приступах
        """
        return self._parse_annotation_file(file_path)
    
    def _parse_bids_format(self, file_path: str) -> List[Dict]:
        """
        Парсинг формата BIDS (Brain Imaging Data Structure)
        
        Параметры:
        file_path (str): путь к файлу разметки в формате BIDS .tsv
        
        Возвращает:
        list: список словарей с информацией о приступах
        """
        seizures = []
        
        # Чтение TSV файла
        df = pd.read_csv(file_path, sep='\t')
        
        # Ожидаемые колонки: onset, duration, trial_type, sample, value
        if 'onset' in df.columns and 'duration' in df.columns:
            for _, row in df.iterrows():
                if pd.notna(row['onset']) and pd.notna(row['duration']):
                    seizure_data = {
                        'start': float(row['onset']),
                        'end': float(row['onset'] + row['duration']),
                        'duration': float(row['duration']),
                        'label': row.get('trial_type', 'seizure')
                    }
                    seizures.append(seizure_data)
        
        return seizures
    
    def _parse_eeglab_format(self, file_path: str) -> List[Dict]:
        """
        Парсинг формата EEGLAB .set файлов (через .fdt или встроенные аннотации)
        
        Параметры:
        file_path (str): путь к файлу разметки в формате EEGLAB
        
        Возвращает:
        list: список словарей с информацией о приступах
        """
        # Для текстовых файлов с разметкой EEGLAB
        seizures = []
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Пропускаем заголовок (если есть)
            start_line = 0
            if lines and lines[0].startswith('Epoch') or lines[0].startswith('Time'):
                start_line = 1
            
            for i in range(start_line, len(lines)):
                line = lines[i].strip()
                if line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            start_time = float(parts[0])
                            end_time = float(parts[1])
                            seizure_data = {
                                'start': start_time,
                                'end': end_time,
                                'duration': end_time - start_time,
                                'label': 'seizure'
                            }
                            seizures.append(seizure_data)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Ошибка при чтении файла EEGLAB {file_path}: {e}")
        
        return seizures
    
    def _parse_custom_format(self, file_path: str) -> List[Dict]:
        """
        Парсинг пользовательского формата разметки
        
        Параметры:
        file_path (str): путь к файлу разметки
        
        Возвращает:
        list: список словарей с информацией о приступах
        """
        # Здесь можно реализовать поддержку специфических форматов
        # Например, JSON, XML или другие текстовые форматы
        
        seizures = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Пример: JSON формат
            if content.strip().startswith('{') or content.strip().startswith('['):
                import json
                data = json.loads(content)
                
                # Ожидаем список словарей с ключами 'start' и 'end'
                if isinstance(data, list):
                    for item in data:
                        if 'start' in item and 'end' in item:
                            seizure_data = {
                                'start': float(item['start']),
                                'end': float(item['end']),
                                'duration': float(item['end'] - item['start']),
                                'label': item.get('label', 'seizure')
                            }
                            seizures.append(seizure_data)
        except Exception as e:
            print(f"Ошибка при чтении пользовательского формата {file_path}: {e}")
        
        return seizures
```

## Интеграция с загрузчиком данных

```python
class IntegratedAnnotationReader(AdvancedSeizureAnnotationReader):
    """
    Модуль для чтения разметки с интеграцией с загрузчиком данных
    """
    
    def __init__(self, data_loader=None):
        """
        Инициализация интегрированного модуля
        
        Параметры:
        data_loader: загрузчик данных (например, EDFLoader)
        """
        super().__init__()
        self.data_loader = data_loader
    
    def get_seizures_for_recording(self, file_id: str, 
                                  recording_duration: Optional[float] = None) -> List[Dict]:
        """
        Получение разметки приступов с проверкой корректности временных меток
        
        Параметры:
        file_id (str): идентификатор файла
        recording_duration (float): длительность записи в секундах
        
        Возвращает:
        list: список словарей с информацией о приступах
        """
        seizures = self.get_seizures(file_id)
        
        # Проверка корректности временных меток
        valid_seizures = []
        
        for seizure in seizures:
            start_time = seizure['start']
            end_time = seizure['end']
            
            # Проверка, что временные метки находятся в пределах записи
            if recording_duration is not None:
                if start_time >= 0 and end_time <= recording_duration and start_time < end_time:
                    valid_seizures.append(seizure)
            else:
                # Если длительность записи не указана, просто проверяем корректность
                if start_time >= 0 and start_time < end_time:
                    valid_seizures.append(seizure)
        
        return valid_seizures
    
    def create_binary_mask(self, file_id: str, 
                          recording_duration: float,
                          sampling_rate: float) -> np.ndarray:
        """
        Создание бинарной маски для разметки приступов
        
        Параметры:
        file_id (str): идентификатор файла
        recording_duration (float): длительность записи в секундах
        sampling_rate (float): частота дискретизации
        
        Возвращает:
        np.ndarray: бинарная маска (1 - приступ, 0 - норма)
        """
        # Создание массива для маски
        num_samples = int(recording_duration * sampling_rate)
        mask = np.zeros(num_samples, dtype=np.int8)
        
        # Получение разметки приступов
        seizures = self.get_seizures_for_recording(file_id, recording_duration)
        
        # Заполнение маски
        for seizure in seizures:
            start_sample = int(seizure['start'] * sampling_rate)
            end_sample = int(seizure['end'] * sampling_rate)
            
            # Ограничение границами массива
            start_sample = max(0, start_sample)
            end_sample = min(num_samples, end_sample)
            
            # Установка меток приступов
            if start_sample < end_sample:
                mask[start_sample:end_sample] = 1
        
        return mask
    
    def get_seizure_windows(self, file_id: str, 
                           window_length: float = 5.0,
                           overlap_ratio: float = 0.0) -> List[Dict]:
        """
        Получение информации о временных окнах с приступами
        
        Параметры:
        file_id (str): идентификатор файла
        window_length (float): длина окна в секундах
        overlap_ratio (float): перекрытие окон (0.0 - 1.0)
        
        Возвращает:
        list: список словарей с информацией об окнах
        """
        # Получение разметки приступов
        seizures = self.get_seizures(file_id)
        
        # Создание списка окон
        windows = []
        
        # Для каждого приступа создаем окна
        for seizure in seizures:
            seizure_start = seizure['start']
            seizure_end = seizure['end']
            seizure_duration = seizure['duration']
            
            # Создание окон, покрывающих приступ
            window_step = window_length * (1.0 - overlap_ratio)
            current_start = max(0, seizure_start - window_length)
            
            while current_start < seizure_end:
                current_end = current_start + window_length
                
                # Проверка пересечения с приступом
                intersection_start = max(current_start, seizure_start)
                intersection_end = min(current_end, seizure_end)
                intersection_duration = max(0, intersection_end - intersection_start)
                
                # Доля времени приступа в окне
                seizure_ratio = intersection_duration / window_length if window_length > 0 else 0
                
                window_info = {
                    'window_start': current_start,
                    'window_end': current_end,
                    'window_duration': window_length,
                    'seizure_ratio': seizure_ratio,
                    'has_seizure': seizure_ratio > 0,
                    'is_full_seizure': seizure_ratio >= 0.9,  # Более 90% окна - приступ
                    'seizure_start_in_window': max(0, seizure_start - current_start),
                    'seizure_end_in_window': min(window_length, seizure_end - current_start)
                }
                
                windows.append(window_info)
                
                # Переход к следующему окну
                current_start += window_step
        
        return windows
```

## Использование модуля

```python
# Пример использования базового модуля
def example_basic_usage():
    """
    Пример использования базового модуля
    """
    # Создание модуля
    reader = SeizureAnnotationReader()
    
    # Загрузка файла разметки
    annotation_file = "ALLratsBaseline/Markers_SWD_BL+H2O/Ati5x1_4April_BL.txt"
    annotation_data = reader.load_annotation_file(annotation_file, "Ati5x1_4April_BL")
    
    print(f"Файл: {annotation_data['file_path']}")
    print(f"Количество приступов: {annotation_data['count']}")
    
    # Получение списка приступов
    seizures = reader.get_seizures("Ati5x1_4April_BL")
    print(f"Первый приступ: начало={seizures[0]['start']}, конец={seizures[0]['end']}")
    
    # Получение разметки в виде DataFrame
    df = reader.get_seizure_dataframe("Ati5x1_4April_BL")
    print(f"DataFrame shape: {df.shape}")
    print(df.head())

# Пример использования расширенного модуля
def example_advanced_usage():
    """
    Пример использования расширенного модуля
    """
    # Создание расширенного модуля
    reader = AdvancedSeizureAnnotationReader()
    
    # Загрузка файла разметки в различных форматах
    files_to_load = [
        ("ALLratsBaseline/Markers_SWD_BL+H2O/Ati5x1_4April_BL.txt", "simple"),
        ("annotations/bids_format.tsv", "bids"),
        ("annotations/eeglab_format.txt", "eeglab")
    ]
    
    for file_path, format_type in files_to_load:
        try:
            annotation_data = reader.load_annotation_file(file_path, format_type=format_type)
            print(f"Загружен файл {file_path} в формате {format_type}")
            print(f"Количество приступов: {annotation_data['count']}")
        except Exception as e:
            print(f"Ошибка при загрузке {file_path}: {e}")
    
    # Получение всей разметки в виде одного DataFrame
    all_seizures_df = reader.get_all_seizures_dataframe()
    print(f"Общее количество приступов: {len(all_seizures_df)}")

# Пример интеграции с загрузчиком данных
def example_integrated_usage():
    """
    Пример использования интегрированного модуля
    """
    # Создание интегрированного модуля
    reader = IntegratedAnnotationReader()
    
    # Загрузка разметки
    annotation_file = "ALLratsBaseline/Markers_SWD_BL+H2O/Ati5x1_4April_BL.txt"
    reader.load_annotation_file(annotation_file, "Ati5x1_4April_BL")
    
    # Создание бинарной маски (предполагаем, что длительность записи 86400 секунд)
    recording_duration = 86400.0  # 24 часа
    sampling_rate = 1000.0  # 1 кГц
    
    mask = reader.create_binary_mask("Ati5x1_4April_BL", recording_duration, sampling_rate)
    print(f"Размер маски: {mask.shape}")
    print(f"Количество точек с приступами: {np.sum(mask)}")
    
    # Получение информации об окнах
    windows = reader.get_seizure_windows("Ati5x1_4April_BL", window_length=5.0, overlap_ratio=0.5)
    print(f"Количество окон: {len(windows)}")
    print(f"Окна с приступами: {sum(1 for w in windows if w['has_seizure'])}")

# Пример пакетной обработки файлов разметки
def batch_processing_example(annotation_dir: str):
    """
    Пример пакетной обработки файлов разметки
    
    Параметры:
    annotation_dir (str): директория с файлами разметки
    """
    import glob
    
    reader = SeizureAnnotationReader()
    
    # Поиск всех файлов разметки
    annotation_files = glob.glob(f"{annotation_dir}/*.txt")
    
    results = []
    
    for file_path in annotation_files:
        try:
            file_id = Path(file_path).stem
            annotation_data = reader.load_annotation_file(file_path, file_id)
            
            results.append({
                'file_path': file_path,
                'file_id': file_id,
                'success': True,
                'seizure_count': annotation_data['count'],
                'total_duration': reader.get_total_seizure_duration(file_id)
            })
            
        except Exception as e:
            results.append({
                'file_path': file_path,
                'file_id': Path(file_path).stem,
                'success': False,
                'error': str(e)
            })
    
    # Создание сводного отчета
    df = pd.DataFrame(results)
    print("Сводный отчет по файлам разметки:")
    print(df[['file_id', 'success', 'seizure_count', 'total_duration']])
    
    return df
```

## Рекомендации по использованию

1. **Выбор формата**: Используйте `AdvancedSeizureAnnotationReader` для поддержки различных форматов разметки
2. **Интеграция**: Используйте `IntegratedAnnotationReader` для интеграции с загрузчиком данных
3. **Проверка корректности**: Всегда проверяйте корректность временных меток и их соответствие длительности записи
4. **Бинарные маски**: Используйте бинарные маски для эффективного представления разметки в числовом виде
5. **Оконная разметка**: Используйте функции для получения информации о временных окнах с приступами

## Возможные улучшения

1. **Поддержка дополнительных форматов**: Добавить поддержку форматов BIDS, EDF annotations, JSON и других
2. **Валидация разметки**: Добавить функции для проверки качества разметки и выявления ошибок
3. **Визуализация**: Добавить функции для визуализации разметки на временных диаграммах
4. **Статистика**: Добавить функции для вычисления статистики по разметке (длительность, частота и т.д.)
5. **Экспорт**: Добавить функции для экспорта разметки в различные форматы (CSV, JSON, BIDS и др.)