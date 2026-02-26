# Стратегия конвертации .adicht в .edf

## Необходимость конвертации

Файлы .adicht (формат ADInstruments LabChart) являются проприетарным форматом, который:
1. Требует специализированного программного обеспечения для чтения
2. Может иметь ограничения по совместимости с различными операционными системами
3. Не поддерживается напрямую большинством библиотек для анализа нейрофизиологических данных

Для унификации обработки данных рекомендуется конвертировать все .adicht файлы в стандартный формат .edf.

## Подходы к конвертации

### Вариант 1: Использование библиотек Python

#### Библиотека neo
```python
import neo

def convert_adicht_to_edf_neo(input_path, output_path):
    """
    Конвертация .adicht файла в .edf с использованием neo
    
    Параметры:
    input_path (str): путь к исходному .adicht файлу
    output_path (str): путь для сохранения .edf файла
    """
    try:
        # Чтение .adicht файла
        reader = neo.io.Spike2IO(filename=input_path)
        block = reader.read_block()
        
        # Запись в .edf формат
        writer = neo.io.EdfIO(filename=output_path)
        writer.write_block(block)
        
        print(f"Файл успешно сконвертирован: {input_path} -> {output_path}")
        return True
    except Exception as e:
        print(f"Ошибка при конвертации {input_path}: {e}")
        return False
```

#### Библиотека mne (через промежуточный формат)
```python
import mne
import numpy as np

def convert_adicht_to_edf_mne(input_path, output_path):
    """
    Конвертация .adicht файла в .edf с использованием mne
    
    Параметры:
    input_path (str): путь к исходному .adicht файлу
    output_path (str): путь для сохранения .edf файла
    """
    try:
        # Чтение данных с помощью neo
        import neo
        reader = neo.io.Spike2IO(filename=input_path)
        block = reader.read_block()
        
        # Извлечение данных
        if len(block.segments) > 0:
            segment = block.segments[0]
            signals = segment.analogsignals
            
            # Подготовка данных для mne
            data = np.array([signal.magnitude.flatten() for signal in signals])
            sfreq = int(signals[0].sampling_rate)
            ch_names = [signal.name for signal in signals]
            
            # Создание RawArray
            info = mne.create_info(ch_names=ch_names, sfreq=sfreq)
            raw = mne.io.RawArray(data, info)
            
            # Сохранение в .edf
            raw.export(output_path, fmt='edf')
            
            print(f"Файл успешно сконвертирован: {input_path} -> {output_path}")
            return True
    except Exception as e:
        print(f"Ошибка при конвертации {input_path}: {e}")
        return False
```

### Вариант 2: Использование внешних инструментов

#### Использование MATLAB (если доступен)
```matlab
function convert_adicht_to_edf_matlab(input_path, output_path)
    % Конвертация .adicht файла в .edf с использованием MATLAB
    try
        % Чтение .adicht файла с помощью ADInstruments SDK
        data = load_adicht_file(input_path);
        
        % Сохранение в .mat формат
        save(output_path_mat, 'data');
        
        % Конвертация в .edf с помощью EEGLAB или других инструментов
        % (требуется дополнительная реализация)
        
        fprintf('Файл успешно сконвертирован: %s -> %s\n', input_path, output_path);
    catch ME
        fprintf('Ошибка при конвертации %s: %s\n', input_path, ME.message);
    end
end
```

## Автоматизированная конвертация всех файлов

```python
import os
import glob
from pathlib import Path

def batch_convert_adicht_files(source_dir, target_dir):
    """
    Пакетная конвертация всех .adicht файлов в директории
    
    Параметры:
    source_dir (str): директория с исходными файлами
    target_dir (str): директория для сохранения сконвертированных файлов
    """
    # Создание директории для результатов
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # Поиск всех .adicht файлов
    adicht_files = glob.glob(os.path.join(source_dir, "**", "*.adicht"), recursive=True)
    
    conversion_log = []
    
    for adicht_file in adicht_files:
        # Определение пути для .edf файла
        relative_path = os.path.relpath(adicht_file, source_dir)
        edf_relative_path = relative_path.replace('.adicht', '.edf')
        edf_file = os.path.join(target_dir, edf_relative_path)
        
        # Создание директории для .edf файла
        Path(os.path.dirname(edf_file)).mkdir(parents=True, exist_ok=True)
        
        # Конвертация файла
        success = convert_adicht_to_edf_neo(adicht_file, edf_file)
        
        # Логирование результата
        conversion_log.append({
            'source': adicht_file,
            'target': edf_file,
            'success': success
        })
        
        if success:
            print(f"Сконвертирован: {adicht_file}")
        else:
            print(f"Ошибка конвертации: {adicht_file}")
    
    # Сохранение лога конвертации
    import json
    log_path = os.path.join(target_dir, 'conversion_log.json')
    with open(log_path, 'w') as f:
        json.dump(conversion_log, f, indent=2)
    
    print(f"Конвертация завершена. Лог сохранен в {log_path}")
    return conversion_log
```

## Создание mapping файла

```python
import pandas as pd

def create_conversion_mapping(conversion_log, output_path):
    """
    Создание таблицы соответствия между исходными и сконвертированными файлами
    
    Параметры:
    conversion_log (list): лог конвертации
    output_path (str): путь для сохранения таблицы
    """
    # Создание DataFrame
    df = pd.DataFrame(conversion_log)
    
    # Фильтрация успешных конвертаций
    successful = df[df['success'] == True]
    
    # Сохранение в CSV
    successful.to_csv(output_path, index=False)
    
    print(f"Mapping файл сохранен: {output_path}")
    print(f"Успешно сконвертировано: {len(successful)} файлов")
    
    return successful
```

## Рекомендации по реализации

1. **Начальный этап**: Попробовать конвертировать несколько файлов вручную для проверки подхода
2. **Тестирование**: Убедиться, что данные сохраняют свою целостность после конвертации
3. **Автоматизация**: Реализовать пакетную конвертацию всех файлов
4. **Документирование**: Создать mapping файл для отслеживания соответствия между исходными и сконвертированными файлами
5. **Резервное копирование**: Перед конвертацией создать резервные копии исходных файлов

## Проверка качества конвертации

```python
def verify_conversion_quality(original_file, converted_file):
    """
    Проверка качества конвертации путем сравнения данных
    
    Параметры:
    original_file (str): путь к исходному файлу
    converted_file (str): путь к сконвертированному файлу
    """
    # Загрузка исходного файла
    original_data = load_adicht_file(original_file)
    
    # Загрузка сконвертированного файла
    converted_data = load_edf_file(converted_file)
    
    # Сравнение параметров
    print(f"Исходный файл: {original_file}")
    print(f"Сконвертированный файл: {converted_file}")
    print(f"Частота дискретизации: {original_data['sampling_freq']} -> {converted_data['sampling_freq']}")
    print(f"Количество каналов: {len(original_data['channel_names'])} -> {len(converted_data['channel_names'])}")
    
    # Сравнение данных (для одного канала)
    if len(original_data['data']) > 0 and len(converted_data['data']) > 0:
        orig_channel = original_data['data'][0]
        conv_channel = converted_data['data'][0]
        
        # Вычисление разницы
        diff = np.mean(np.abs(orig_channel - conv_channel[:len(orig_channel)]))
        print(f"Средняя разница данных: {diff}")
```

## Заключение

Конвертация .adicht файлов в .edf позволит унифицировать обработку данных и использовать стандартные библиотеки для анализа нейрофизиологических сигналов. Рекомендуется начать с тестирования подхода на нескольких файлах, а затем перейти к пакетной конвертации всего датасета.