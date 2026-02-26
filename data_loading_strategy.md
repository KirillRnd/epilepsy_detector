# Стратегия чтения файлов .adicht и .edf

## Обработка .edf файлов

### Используемые библиотеки
1. **mne** - основная библиотека для обработки нейрофизиологических данных
   - `mne.io.read_raw_edf()` - основная функция для чтения .edf файлов
   - Поддерживает различные параметры предобработки при чтении

2. **pyedflib** - низкоуровневая библиотека для работы с EDF файлами
   - Более быстрая, но менее функциональная по сравнению с mne
   - Подходит для простого чтения данных

### Рекомендуемый подход
Использовать **mne** как основную библиотеку, так как она:
- Поддерживает широкий спектр функций для предобработки
- Имеет хорошую документацию и сообщество
- Позволяет легко работать с аннотациями и метаданными

### Базовый загрузчик для .edf файлов

```python
import mne
import numpy as np

def load_edf_file(file_path, preload=True):
    """
    Загрузка .edf файла с помощью mne
    
    Параметры:
    file_path (str): путь к .edf файлу
    preload (bool): загружать данные в память сразу
    
    Возвращает:
    raw (mne.io.Raw): объект с сырыми данными
    """
    # Загрузка файла
    raw = mne.io.read_raw_edf(file_path, preload=preload)
    
    # Получение информации
    sampling_freq = raw.info['sfreq']  # частота дискретизации
    channel_names = raw.ch_names        # имена каналов
    data = raw.get_data()               # данные в виде numpy массива
    
    return {
        'raw': raw,
        'data': data,
        'sampling_freq': sampling_freq,
        'channel_names': channel_names
    }
```

## Обработка .adicht файлов (LabChart)

### Особенности формата
- Файлы .adicht создаются программой ADInstruments LabChart
- Это проприетарный формат, требующий специализированных библиотек для чтения
- Может содержать несколько каналов с разными частотами дискретизации

### Стратегии обработки

#### Вариант 1: Использование специализированных библиотек
1. **neo** - библиотека для чтения нейрофизиологических форматов
   - Поддерживает чтение .adicht файлов через io модуль
   - Может потребоваться установка дополнительных зависимостей

2. **scipy** - для чтения через промежуточные форматы
   - Если файлы предварительно конвертированы в .mat

#### Вариант 2: Конвертация в стандартные форматы
Рекомендуемый подход:
1. Конвертировать все .adicht файлы в .edf (или .mat) с помощью LabChart
2. Далее работать с единым форматом .edf

### Реализация конвертации
Поскольку у нас нет прямого доступа к LabChart, мы будем использовать следующий подход:
1. Использовать neo для попытки чтения .adicht файлов
2. Если это не работает, подготовить скрипты для конвертации

### Базовый загрузчик для .adicht файлов

```python
import neo

def load_adicht_file(file_path):
    """
    Загрузка .adicht файла с помощью neo
    
    Параметры:
    file_path (str): путь к .adicht файлу
    
    Возвращает:
    data (dict): словарь с данными и метаданными
    """
    try:
        # Создание объекта для чтения
        reader = neo.io.Spike2IO(filename=file_path)
        
        # Чтение блока данных
        block = reader.read_block()
        
        # Извлечение данных
        segments = block.segments
        if len(segments) > 0:
            segment = segments[0]  # Берем первый сегмент
            analog_signals = segment.analogsignals
            
            # Преобразование в numpy массивы
            signals_data = [signal.magnitude for signal in analog_signals]
            sampling_rates = [signal.sampling_rate for signal in analog_signals]
            channel_names = [signal.name for signal in analog_signals]
            
            return {
                'signals': signals_data,
                'sampling_rates': sampling_rates,
                'channel_names': channel_names,
                'segments': len(segments)
            }
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
        return None
```

## Единый интерфейс для загрузки файлов

```python
def load_recording_file(file_path):
    """
    Универсальная функция для загрузки файлов .edf и .adicht
    
    Параметры:
    file_path (str): путь к файлу
    
    Возвращает:
    data (dict): словарь с данными и метаданными
    """
    if file_path.endswith('.edf'):
        return load_edf_file(file_path)
    elif file_path.endswith('.adicht'):
        return load_adicht_file(file_path)
    else:
        raise ValueError(f"Неподдерживаемый формат файла: {file_path}")
```

## Рекомендации по реализации

1. Начать с реализации загрузчика для .edf файлов с использованием mne
2. Протестировать чтение нескольких файлов из датасета
3. Попробовать реализовать загрузчик для .adicht файлов с помощью neo
4. При необходимости подготовить скрипты для конвертации .adicht в .edf
5. Создать единый интерфейс для загрузки файлов обоих форматов