import pandas as pd
import os
from pathlib import Path

def analyze_excel_structure(file_path):
    """
    Анализ структуры Excel файла: листы и столбцы
    """
    print(f"Анализ файла: {file_path}")
    
    # Проверяем существование файла
    if not os.path.exists(file_path):
        print(f"Файл {file_path} не найден")
        return
    
    # Читаем файл Excel
    try:
        # Получаем список листов
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        print(f"Найдено листов: {len(sheet_names)}")
        print("Список листов:")
        for i, sheet in enumerate(sheet_names):
            print(f"  {i+1}. {sheet}")
            
            # Читаем первую строку листа (названия столбцов)
            df = pd.read_excel(file_path, sheet_name=sheet, nrows=5)  # Читаем первые 5 строк для анализа
            columns = df.columns.tolist()
            print(f"    Столбцы ({len(columns)}): {columns}")
            
            # Выводим информацию о данных
            print(f"    Размер данных: {df.shape[0]} строк x {df.shape[1]} столбцов")
            print(f"    Типы данных:")
            for col in df.columns:
                print(f"      {col}: {df[col].dtype}")
            
            # Показываем пример данных
            print(f"    Пример данных:")
            print(df.head(3))
            print()
            
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")

def main():
    file_path = "ALLratsBaseline/Markers_SWD_BL+H2O/Ati-Dex_SWD_ind.xlsx"
    
    # Анализ структуры файла
    analyze_excel_structure(file_path)

if __name__ == "__main__":
    main()