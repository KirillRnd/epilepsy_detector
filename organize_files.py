import os
import re
import shutil

def organize_files():
    # Создаем словарь для хранения файлов по идентификаторам животных
    animal_files = {}
    
    # Проходим по всем файлам в data/raw
    for root, dirs, files in os.walk('data/raw'):
        for file in files:
            if file.endswith(('.adicht', '.edf', '.txt')):
                # Извлекаем идентификатор животного из имени файла
                match = re.match(r'([A-Za-z0-9]+)', file)
                if match:
                    animal_id = match.group(1)
                    
                    # Создаем список файлов для каждого животного
                    if animal_id not in animal_files:
                        animal_files[animal_id] = []
                    animal_files[animal_id].append(os.path.join(root, file))
    
    # Организуем файлы по папкам
    for animal_id, files in animal_files.items():
        # Создаем папку для животного
        animal_dir = os.path.join('data', 'raw', animal_id)
        os.makedirs(animal_dir, exist_ok=True)
        
        # Перемещаем файлы в папку животного
        for file_path in files:
            filename = os.path.basename(file_path)
            new_path = os.path.join(animal_dir, filename)
            
            # Если файл уже существует, добавляем суффикс
            counter = 1
            base_name, ext = os.path.splitext(filename)
            while os.path.exists(new_path):
                new_name = f"{base_name}_{counter}{ext}"
                new_path = os.path.join(animal_dir, new_name)
                counter += 1
            
            shutil.move(file_path, new_path)
    
    # Удаляем пустые папки
    for root, dirs, files in os.walk('data/raw', topdown=False):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
            except OSError:
                pass

if __name__ == "__main__":
    organize_files()
    print("Файлы успешно организованы по папкам животных")