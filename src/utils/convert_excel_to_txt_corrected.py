import pandas as pd
import os
import re

def sanitize_filename(name):
    """Очищает имя файла от недопустимых символов"""
    # Заменяем недопустимые символы на подчеркивания
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    # Убираем лишние пробелы
    name = re.sub(r'\s+', '_', name.strip())
    return name

def convert_excel_to_txt_corrected(excel_file, output_dir):
    """
    Преобразует Excel файл в несколько .txt файлов с корректным форматом
    
    Args:
        excel_file (str): Путь к Excel файлу
        output_dir (str): Директория для сохранения .txt файлов
    """
    # Создаем директорию для вывода, если она не существует
    os.makedirs(output_dir, exist_ok=True)
    
    # Читаем Excel файл
    xls = pd.ExcelFile(excel_file)
    
    # Счетчик созданных файлов
    file_count = 0
    
    # Обрабатываем каждый лист
    for sheet_name in xls.sheet_names:
        print(f"Обработка листа: {sheet_name}")
        
        # Читаем лист без заголовков для корректной обработки
        df = pd.read_excel(excel_file, sheet_name=sheet_name, header=None)
        
        # Получаем первую строку с названиями условий
        header_row = df.iloc[0]
        
        # Определяем группы условий
        condition_groups = []
        i = 0
        
        while i < len(header_row):
            # Проверяем, есть ли название условия в текущем столбце
            condition_name = header_row.iloc[i] if i < len(header_row) else None
            
            # Проверяем, что это действительно название условия (не NaN и строка)
            if pd.notna(condition_name) and isinstance(condition_name, str) and condition_name.strip():
                # Проверяем, что следующие 4 столбца существуют
                if i + 4 <= len(df.columns):
                    # Используем правильные индексы для столбцов начала и конца
                    start_col = i + 1
                    end_col = i + 2
                    
                    if start_col < len(df.columns) and end_col < len(df.columns):
                        # Проверяем, что столбцы содержат числовые данные
                        if df.iloc[1:, start_col].dtype in ['int64', 'float64'] and df.iloc[1:, end_col].dtype in ['int64', 'float64']:
                            condition_groups.append((condition_name, i, i+1, start_col, end_col))
                            print(f"  Найдена группа условий: {condition_name}")
            
            # Переходим к следующей возможной группе (через 5 столбцов)
            i += 5
        
        # Обрабатываем каждую группу условий
        for condition_name, number_col, _, start_col, end_col in condition_groups:
            # Создаем имя файла
            filename = f"{sheet_name}_{sanitize_filename(condition_name)}.txt"
            filepath = os.path.join(output_dir, filename)
            
            # Извлекаем данные для этой группы
            data_rows = []
            
            # Проходим по всем строкам с данными (начиная со второй строки)
            for idx in range(1, len(df)):
                start_time = df.iloc[idx, start_col] if start_col < len(df.columns) else None
                end_time = df.iloc[idx, end_col] if end_col < len(df.columns) else None
                
                # Проверяем, что значения существуют и являются числами
                if pd.notna(start_time) and pd.notna(end_time) and isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    # Форматируем значения с запятой как десятичным разделителем
                    # Округляем до 1 знака после запятой, как в примерах
                    if isinstance(start_time, float):
                        start_str = f"{start_time:.1f}".replace('.', ',')
                    else:
                        start_str = str(start_time).replace('.', ',')
                    
                    if isinstance(end_time, float):
                        end_str = f"{end_time:.1f}".replace('.', ',')
                    else:
                        end_str = str(end_time).replace('.', ',')
                    
                    data_rows.append(f"{start_str}\t{end_str}")
            
            # Записываем данные в файл
            if data_rows:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(data_rows))
                print(f"  Создан файл: {filename} с {len(data_rows)} записями")
                file_count += 1
            else:
                print(f"  Файл {filename} не создан - нет данных")
    
    print(f"\nОбработка завершена. Создано файлов: {file_count}")

# Запуск преобразования
if __name__ == "__main__":
    excel_file = "ALLratsBaseline/Markers_SWD_BL+H2O/Ati-Dex_SWD_ind.xlsx"
    output_dir = "ALLratsBaseline/Markers_SWD_BL+H2O/converted_corrected"
    
    convert_excel_to_txt_corrected(excel_file, output_dir)
    print(f"\nВсе файлы сохранены в директории: {output_dir}")