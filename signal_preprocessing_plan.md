# План предобработки сигналов ЭЭГ

## Общая архитектура предобработки

Предобработка сигналов ЭЭГ включает несколько этапов:
1. Фильтрация сигналов
2. Ресемплинг к единой частоте дискретизации
3. Нормализация амплитуд
4. Обнаружение и обработка артефактов
5. Сегментация сигналов на временные окна

## 1. Фильтрация сигналов

### Полосовой фильтр
```python
def apply_bandpass_filter(data: np.ndarray, 
                         low_freq: float = 1.0,
                         high_freq: float = 100.0,
                         sampling_rate: float = 1000.0,
                         filter_order: int = 4) -> np.ndarray:
    """
    Применение полосового фильтра к сигналу
    
    Параметры:
    data (np.ndarray): входные данные (каналы × время)
    low_freq (float): нижняя частота среза
    high_freq (float): верхняя частота среза
    sampling_rate (float): частота дискретизации
    filter_order (int): порядок фильтра
    
    Возвращает:
    np.ndarray: отфильтрованные данные
    """
    from scipy import signal
    
    # Нормализация частот
    nyquist = sampling_rate / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Создание фильтра
    b, a = signal.butter(filter_order, [low_norm, high_norm], btype='band')
    
    # Применение фильтра (zero-phase)
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    
    return filtered_data
```

### Режекторный фильтр (подавление сетевых помех)
```python
def apply_notch_filter(data: np.ndarray,
                      notch_freq: float = 50.0,
                      sampling_rate: float = 1000.0,
                      quality_factor: float = 30.0) -> np.ndarray:
    """
    Применение режекторного фильтра для подавления сетевых помех
    
    Параметры:
    data (np.ndarray): входные данные
    notch_freq (float): частота режекторного фильтра (Гц)
    sampling_rate (float): частота дискретизации
    quality_factor (float): коэффициент качества фильтра
    
    Возвращает:
    np.ndarray: данные после применения режекторного фильтра
    """
    from scipy import signal
    
    # Создание режекторного фильтра
    b, a = signal.iirnotch(notch_freq, quality_factor, sampling_rate)
    
    # Применение фильтра (zero-phase)
    filtered_data = signal.filtfilt(b, a, data, axis=1)
    
    return filtered_data
```

## 2. Ресемплинг к единой частоте дискретизации

```python
def resample_signal(data: np.ndarray,
                   original_rate: float,
                   target_rate: float) -> np.ndarray:
    """
    Ресемплинг сигнала к целевой частоте дискретизации
    
    Параметры:
    data (np.ndarray): входные данные
    original_rate (float): исходная частота дискретизации
    target_rate (float): целевая частота дискретизации
    
    Возвращает:
    np.ndarray: ресемплированные данные
    """
    from scipy import signal
    import numpy as np
    
    # Вычисление коэффициента ресемплинга
    resample_ratio = target_rate / original_rate
    target_length = int(data.shape[1] * resample_ratio)
    
    # Ресемплинг
    resampled_data = signal.resample(data, target_length, axis=1)
    
    return resampled_data
```

## 3. Нормализация амплитуд

### Z-нормализация
```python
def z_normalize(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Z-нормализация данных (вычитание среднего и деление на стандартное отклонение)
    
    Параметры:
    data (np.ndarray): входные данные
    axis (int): ось по которой производится нормализация
    
    Возвращает:
    np.ndarray: нормализованные данные
    """
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    
    # Избегаем деления на ноль
    normalized_data = (data - mean) / (std + 1e-8)
    
    return normalized_data
```

### Минимакс нормализация
```python
def minmax_normalize(data: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    Минимакс нормализация данных (приведение к диапазону [0, 1])
    
    Параметры:
    data (np.ndarray): входные данные
    axis (int): ось по которой производится нормализация
    
    Возвращает:
    np.ndarray: нормализованные данные
    """
    min_val = np.min(data, axis=axis, keepdims=True)
    max_val = np.max(data, axis=axis, keepdims=True)
    
    # Избегаем деления на ноль
    normalized_data = (data - min_val) / (max_val - min_val + 1e-8)
    
    return normalized_data
```

## 4. Обнаружение и обработка артефактов

### Обнаружение артефактов по амплитуде
```python
def detect_amplitude_artifacts(data: np.ndarray,
                              threshold: float = 5.0,
                              window_size: int = 100) -> np.ndarray:
    """
    Обнаружение артефактов по превышению порога амплитуды
    
    Параметры:
    data (np.ndarray): входные данные
    threshold (float): порог амплитуды (кратный стандартному отклонению)
    window_size (int): размер окна для вычисления статистики
    
    Возвращает:
    np.ndarray: бинарная маска артефактов
    """
    # Вычисление стандартного отклонения по окнам
    std_values = np.array([
        np.std(data[:, i:i+window_size], axis=1) 
        for i in range(0, data.shape[1], window_size)
    ]).T
    
    # Вычисление среднего стандартного отклонения по каналам
    mean_std = np.mean(std_values, axis=0, keepdims=True)
    
    # Обнаружение артефактов
    artifact_mask = np.abs(data) > (threshold * mean_std)
    
    # Объединение по каналам (если артефакт в любом канале)
    artifact_mask = np.any(artifact_mask, axis=0)
    
    return artifact_mask
```

### Удаление артефактов
```python
def remove_artifacts(data: np.ndarray,
                    artifact_mask: np.ndarray,
                    method: str = 'zero') -> np.ndarray:
    """
    Удаление артефактов из данных
    
    Параметры:
    data (np.ndarray): входные данные
    artifact_mask (np.ndarray): бинарная маска артефактов
    method (str): метод удаления ('zero', 'interpolate', 'reject')
    
    Возвращает:
    np.ndarray: данные без артефактов
    """
    if method == 'zero':
        # Замена артефактов нулями
        cleaned_data = data.copy()
        cleaned_data[:, artifact_mask] = 0
        
    elif method == 'interpolate':
        # Интерполяция артефактов
        cleaned_data = data.copy()
        artifact_indices = np.where(artifact_mask)[0]
        
        for ch in range(data.shape[0]):
            channel_data = cleaned_data[ch, :]
            # Простая линейная интерполяция
            for idx in artifact_indices:
                if idx > 0 and idx < len(channel_data) - 1:
                    channel_data[idx] = (channel_data[idx-1] + channel_data[idx+1]) / 2
            cleaned_data[ch, :] = channel_data
            
    elif method == 'reject':
        # Удаление окон с артефактами (возвращает маску для сохранения)
        cleaned_data = data.copy()
        
    return cleaned_data
```

## 5. Сегментация сигналов на временные окна

```python
def create_sliding_windows(data: np.ndarray,
                          window_length: int,
                          step_size: int,
                          overlap_ratio: float = 0.0) -> np.ndarray:
    """
    Создание скользящих окон из сигнала
    
    Параметры:
    data (np.ndarray): входные данные (каналы × время)
    window_length (int): длина окна в отсчетах
    step_size (int): шаг между окнами в отсчетах
    overlap_ratio (float): перекрытие окон (0.0 - 1.0)
    
    Возвращает:
    np.ndarray: массив окон (окна × каналы × время)
    """
    if overlap_ratio > 0:
        step_size = int(window_length * (1.0 - overlap_ratio))
    
    # Вычисление количества окон
    num_windows = (data.shape[1] - window_length) // step_size + 1
    
    # Создание окон
    windows = np.zeros((num_windows, data.shape[0], window_length))
    
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_length
        windows[i, :, :] = data[:, start_idx:end_idx]
    
    return windows
```

## Комплексная предобработка

```python
class SignalPreprocessor:
    """
    Комплексный класс для предобработки сигналов ЭЭГ
    """
    
    def __init__(self, 
                 target_sampling_rate: float = 512.0,
                 bandpass_low: float = 1.0,
                 bandpass_high: float = 100.0,
                 notch_freq: float = 50.0,
                 normalize_method: str = 'z_score'):
        """
        Инициализация предобработчика
        
        Параметры:
        target_sampling_rate (float): целевая частота дискретизации
        bandpass_low (float): нижняя частота полосового фильтра
        bandpass_high (float): верхняя частота полосового фильтра
        notch_freq (float): частота режекторного фильтра
        normalize_method (str): метод нормализации ('z_score', 'minmax')
        """
        self.target_sampling_rate = target_sampling_rate
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.normalize_method = normalize_method
    
    def preprocess(self, data: np.ndarray, 
                   original_sampling_rate: float,
                   artifact_removal: bool = True) -> Dict:
        """
        Комплексная предобработка сигнала
        
        Параметры:
        data (np.ndarray): входные данные (каналы × время)
        original_sampling_rate (float): исходная частота дискретизации
        artifact_removal (bool): удалять артефакты
        
        Возвращает:
        dict: словарь с предобработанными данными и метаданными
        """
        preprocessing_steps = []
        
        # 1. Ресемплинг
        if original_sampling_rate != self.target_sampling_rate:
            data = resample_signal(data, original_sampling_rate, self.target_sampling_rate)
            preprocessing_steps.append('resampling')
        
        # 2. Полосовой фильтр
        data = apply_bandpass_filter(
            data, 
            self.bandpass_low, 
            self.bandpass_high, 
            self.target_sampling_rate
        )
        preprocessing_steps.append('bandpass_filter')
        
        # 3. Режекторный фильтр
        data = apply_notch_filter(data, self.notch_freq, self.target_sampling_rate)
        preprocessing_steps.append('notch_filter')
        
        # 4. Обнаружение и удаление артефактов
        if artifact_removal:
            artifact_mask = detect_amplitude_artifacts(data)
            data = remove_artifacts(data, artifact_mask, method='interpolate')
            preprocessing_steps.append('artifact_removal')
        
        # 5. Нормализация
        if self.normalize_method == 'z_score':
            data = z_normalize(data)
        elif self.normalize_method == 'minmax':
            data = minmax_normalize(data)
        preprocessing_steps.append('normalization')
        
        return {
            'data': data,
            'sampling_rate': self.target_sampling_rate,
            'steps': preprocessing_steps,
            'artifact_mask': artifact_mask if artifact_removal else None
        }
    
    def preprocess_window(self, window: np.ndarray) -> np.ndarray:
        """
        Предобработка отдельного временного окна
        
        Параметры:
        window (np.ndarray): временное окно (каналы × время)
        
        Возвращает:
        np.ndarray: предобработанное окно
        """
        # Нормализация окна
        if self.normalize_method == 'z_score':
            window = z_normalize(window)
        elif self.normalize_method == 'minmax':
            window = minmax_normalize(window)
        
        return window
```

## Параметры предобработки для проекта детектирования эпилепсии

### Рекомендуемые параметры:
1. **Частота дискретизации**: 512 Гц (компромисс между качеством и производительностью)
2. **Полосовой фильтр**: 1-100 Гц (основной диапазон ЭЭГ)
3. **Режекторный фильтр**: 50 Гц (сетевые помехи)
4. **Нормализация**: Z-нормализация
5. **Обработка артефактов**: обнаружение и интерполяция

### Конфигурация предобработки:
```python
PREPROCESSING_CONFIG = {
    'target_sampling_rate': 512.0,
    'bandpass_low': 1.0,
    'bandpass_high': 100.0,
    'notch_freq': 50.0,
    'normalize_method': 'z_score',
    'artifact_removal': True,
    'artifact_threshold': 5.0,
    'window_length': 5.0,  # секунды
    'window_overlap': 0.5   # 50% перекрытие
}
```

## Валидация предобработки

```python
def validate_preprocessing(original_data: np.ndarray,
                         preprocessed_data: np.ndarray,
                         config: Dict) -> Dict:
    """
    Валидация результатов предобработки
    
    Параметры:
    original_data (np.ndarray): исходные данные
    preprocessed_data (np.ndarray): предобработанные данные
    config (dict): конфигурация предобработки
    
    Возвращает:
    dict: результаты валидации
    """
    validation_results = {
        'original_shape': original_data.shape,
        'preprocessed_shape': preprocessed_data.shape,
        'sampling_rate': config['target_sampling_rate'],
        'data_range': [np.min(preprocessed_data), np.max(preprocessed_data)],
        'mean': np.mean(preprocessed_data),
        'std': np.std(preprocessed_data),
        'has_nans': np.isnan(preprocessed_data).any(),
        'has_infs': np.isinf(preprocessed_data).any()
    }
    
    return validation_results
```

## Рекомендации по реализации

1. **Последовательность этапов**: Строго соблюдать последовательность предобработки
2. **Сохранение метаданных**: Всегда сохранять информацию о примененных этапах
3. **Кэширование**: Кэшировать предобработанные данные для ускорения повторного использования
4. **Проверка качества**: Всегда проверять качество предобработки визуально на примерах
5. **Гибкость**: Предоставлять возможность настройки параметров для разных типов данных

## Возможные улучшения

1. **Адаптивная фильтрация**: Использовать адаптивные фильтры для подавления артефактов
2. **Независимые компоненты**: Применение ICA для разделения источников сигналов
3. **Автоматическая настройка**: Автоматическая настройка параметров фильтров под конкретные данные
4. **Параллельная обработка**: Реализация параллельной обработки для ускорения
5. **Логирование**: Подробное логирование всех этапов предобработки