# Шаблон метаданных для файлов ЭЭГ

## Описание полей таблицы метаданных

| Поле | Тип данных | Описание |
|------|------------|----------|
| animal_id | string | Идентификатор животного (например, Dex1y2) |
| session_id | string | Идентификатор сессии (например, BL_10May) |
| file_path | string | Относительный путь к файлу |
| file_format | string | Формат файла (edf, adicht) |
| recording_date | date | Дата записи |
| cycle_phase | string | Фаза цикла (эструс, диэструс, метэструс, проэструс) |
| condition | string | Условие эксперимента (BL - базовая линия, H2O - контроль) |
| body_location | string | Место установки электродов (голова, спина, хвост и т.д.) |
| body_weight | float | Вес животного в граммах |
| age_days | float | Возраст животного в днях |
| sampling_rate | float | Частота дискретизации (Гц) |
| num_channels | int | Количество каналов |
| duration_sec | float | Длительность записи (секунды) |
| has_seizures | boolean | Наличие эпилептических приступов |
| notes | string | Дополнительные заметки |

## Пример заполнения таблицы

| animal_id | session_id | file_path | file_format | recording_date | cycle_phase | condition | body_location | body_weight | age_days | sampling_rate | num_channels | duration_sec | has_seizures | notes |
|-----------|------------|-----------|--------------|----------------|--------------|-----------|----------------|-------------|----------|---------------|--------------|--------------|---------------|-------|
| Dex1y2 | BL_10May | data/raw/Dex1y2/BL_10May/Dex1y2_10May_BL.edf | edf | 2023-05-10 | не указано | BL | спина | 372.0 | 11.2 | 1000.0 | 4 | 86400.0 | false | Крыса без эпилепсии |
| Ati5y1 | BL_22Mart | data/raw/Ati5y1/BL_22Mart/Ati5y1_22Mart_BL_24h.edf | edf | 2023-03-22 | не указано | BL | голова | 356.0 | 9.4 | 1000.0 | 4 | 86400.0 | true | Мама Ati5 с сильной аудиогенной эпилепсией |

## Скрипт для создания метаданных

```python
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

def create_metadata_table(source_dir, output_path):
    """
    Создание таблицы метаданных для всех файлов ЭЭГ
    
    Параметры:
    source_dir (str): директория с исходными файлами
    output_path (str): путь для сохранения таблицы метаданных
    """
    # Список для хранения метаданных
    metadata_list = []
    
    # Проход по всем файлам в директории
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith(('.edf', '.adicht')):
                # Получение пути к файлу
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, source_dir)
                
                # Извлечение информации из имени файла и пути
                file_info = extract_file_info(file, relative_path, root)
                
                # Добавление в список
                metadata_list.append(file_info)
    
    # Создание DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Сохранение в CSV
    df.to_csv(output_path, index=False)
    
    print(f"Таблица метаданных создана: {output_path}")
    print(f"Всего файлов: {len(df)}")
    
    return df

def extract_file_info(filename, relative_path, full_path):
    """
    Извлечение информации из имени файла и пути
    
    Параметры:
    filename (str): имя файла
    relative_path (str): относительный путь
    full_path (str): полный путь
    
    Возвращает:
    dict: словарь с метаданными файла
    """
    # Базовая информация
    file_info = {
        'animal_id': '',
        'session_id': '',
        'file_path': relative_path,
        'file_format': 'edf' if filename.endswith('.edf') else 'adicht',
        'recording_date': '',
        'cycle_phase': '',
        'condition': '',
        'body_location': '',
        'body_weight': '',
        'age_days': '',
        'sampling_rate': '',
        'num_channels': '',
        'duration_sec': '',
        'has_seizures': '',
        'notes': ''
    }
    
    # Извлечение animal_id и session_id из пути
    path_parts = relative_path.split(os.sep)
    if len(path_parts) >= 3:
        file_info['animal_id'] = path_parts[0]
        file_info['session_id'] = path_parts[1]
    
    # Извлечение информации из имени файла
    base_name = os.path.splitext(filename)[0]
    
    # Попытка извлечь дату из имени файла
    date_patterns = ['%d%b%Y', '%d%B%Y', '%Y-%m-%d', '%d%m%Y']
    for pattern in date_patterns:
        try:
            # Пример: извлечение даты из "Dex1y2_10May_BL"
            parts = base_name.split('_')
            for part in parts:
                if any(char.isdigit() for char in part):
                    # Попытка распознать дату
                    date_obj = datetime.strptime(part, pattern)
                    file_info['recording_date'] = date_obj.strftime('%Y-%m-%d')
                    break
        except ValueError:
            continue
    
    # Определение условия эксперимента
    if 'BL' in base_name.upper():
        file_info['condition'] = 'BL'
    elif 'H2O' in base_name.upper():
        file_info['condition'] = 'H2O'
    
    # Определение фазы цикла (если указана)
    cycle_keywords = {
        'эструс': 'эструс',
        'estrus': 'эструс',
        'диэструс': 'диэструс',
        'diestr': 'диэструс',
        'метэструс': 'метэструс',
        'проэструс': 'проэструс',
        'proestr': 'проэструс'
    }
    
    for keyword, phase in cycle_keywords.items():
        if keyword in base_name.lower():
            file_info['cycle_phase'] = phase
            break
    
    return file_info

# Пример использования
if __name__ == "__main__":
    source_directory = "ALLratsBaseline"
    output_file = "data_metadata.csv"
    
    metadata_df = create_metadata_table(source_directory, output_file)
    
    # Вывод статистики
    print("\nСтатистика по форматам файлов:")
    print(metadata_df['file_format'].value_counts())
    
    print("\nСтатистика по условиям:")
    print(metadata_df['condition'].value_counts())
```

## Дополнительные скрипты для обогащения метаданных

```python
def enrich_metadata_with_readme(metadata_df, readme_path):
    """
    Обогащение метаданных информацией из Read_me_BL+H2O.xlsx
    
    Параметры:
    metadata_df (DataFrame): таблица метаданных
    readme_path (str): путь к файлу Read_me_BL+H2O.xlsx
    
    Возвращает:
    DataFrame: обогащенная таблица метаданных
    """
    # Чтение файла Read_me_BL+H2O.xlsx
    readme_df = pd.read_excel(readme_path)
    
    # Объединение данных
    # (требуется реализация логики сопоставления)
    
    return metadata_df

def extract_sampling_rate_and_channels(file_path):
    """
    Извлечение частоты дискретизации и количества каналов из файла
    
    Параметры:
    file_path (str): путь к файлу
    
    Возвращает:
    dict: словарь с параметрами
    """
    try:
        if file_path.endswith('.edf'):
            import mne
            raw = mne.io.read_raw_edf(file_path, preload=False)
            return {
                'sampling_rate': raw.info['sfreq'],
                'num_channels': len(raw.ch_names),
                'channel_names': raw.ch_names
            }
        elif file_path.endswith('.adicht'):
            # Для .adicht файлов требуется специальная обработка
            # (реализация зависит от доступных библиотек)
            pass
    except Exception as e:
        print(f"Ошибка при извлечении параметров из {file_path}: {e}")
        return None
```

## Рекомендации по заполнению метаданных

1. **Автоматическое извлечение**: Использовать скрипты для автоматического извлечения базовой информации из имен файлов и путей
2. **Ручное заполнение**: Дополнить автоматически извлеченную информацию данными из Read_me_BL+H2O.xlsx
3. **Проверка данных**: Проверить корректность извлеченной информации и при необходимости внести корректировки
4. **Обновление**: Регулярно обновлять метаданные при добавлении новых файлов в датасет