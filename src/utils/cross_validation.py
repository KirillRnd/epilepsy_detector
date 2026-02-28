import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple
import os
import json

class AnimalCrossValidation:
    """
    Класс для реализации кросс-валидации по животным
    """
    
    def __init__(self, data_dir: str, n_splits: int = 5, seed: int = 42):
        """
        Инициализация кросс-валидации
        
        Параметры:
        data_dir (str): директория с предобработанными данными
        n_splits (int): количество фолдов
        seed (int): seed для воспроизводимости
        """
        self.data_dir = Path(data_dir)
        self.n_splits = n_splits
        self.seed = seed
        np.random.seed(seed)
        
        # Получение списка уникальных животных
        self.animal_ids = self._get_animal_ids()
    
    def _get_animal_ids(self) -> List[str]:
        """
        Получение списка идентификаторов животных
        
        Возвращает:
        list: список идентификаторов животных
        """
        animal_ids = []
        for animal_dir in self.data_dir.iterdir():
            if animal_dir.is_dir():
                animal_ids.append(animal_dir.name)
        return sorted(animal_ids)
    
    def get_splits(self) -> List[Tuple[List[str], List[str]]]:
        """
        Получение разбиения на фолды для кросс-валидации
        
        Возвращает:
        list: список кортежей (train_animals, val_animals) для каждого фолда
        """
        # Перемешивание животных
        shuffled_animals = self.animal_ids.copy()
        np.random.shuffle(shuffled_animals)
        
        # Разбиение на фолды
        fold_size = len(shuffled_animals) // self.n_splits
        splits = []
        
        for i in range(self.n_splits):
            # Определение валидационных животных для текущего фолда
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else len(shuffled_animals)
            val_animals = shuffled_animals[val_start:val_end]
            
            # Определение обучающих животных
            train_animals = [animal for animal in shuffled_animals if animal not in val_animals]
            
            splits.append((train_animals, val_animals))
        
        return splits
    
    def save_splits(self, output_dir: str):
        """
        Сохранение разбиения на фолды в файл
        
        Параметры:
        output_dir (str): директория для сохранения
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        splits = self.get_splits()
        
        # Сохранение информации о фолдах
        splits_info = {
            'n_splits': self.n_splits,
            'seed': self.seed,
            'folds': []
        }
        
        for i, (train_animals, val_animals) in enumerate(splits):
            fold_info = {
                'fold': i,
                'train_animals': train_animals,
                'val_animals': val_animals
            }
            splits_info['folds'].append(fold_info)
        
        # Сохранение в JSON файл
        splits_file = output_path / 'cv_splits.json'
        with open(splits_file, 'w', encoding='utf-8') as f:
            json.dump(splits_info, f, ensure_ascii=False, indent=2)
        
        print(f"Разбиение на фолды сохранено в: {splits_file}")
        return splits_file

def run_cross_validation(config_path: str, n_splits: int = 5):
    """
    Запуск кросс-валидации
    
    Параметры:
    config_path (str): путь к файлу конфигурации
    n_splits (int): количество фолдов
    """
    import yaml
    import sys
    import os
    
    # Добавляем путь к модулям проекта
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(project_root)
    
    # Загрузка конфигурации
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Создание объекта кросс-валидации
    cv = AnimalCrossValidation(
        data_dir=config['data']['data_dir'],
        n_splits=n_splits,
        seed=config['experiment']['seed']
    )
    
    # Получение разбиения на фолды
    splits = cv.get_splits()
    
    # Запуск обучения для каждого фолда
    fold_results = []
    
    for fold_idx, (train_animals, val_animals) in enumerate(splits):
        print(f"\n=== Фолд {fold_idx + 1}/{n_splits} ===")
        print(f"Обучающие животные: {train_animals}")
        print(f"Валидационные животные: {val_animals}")
        
        # Здесь должен быть код для запуска обучения с конкретным разбиением
        # В реальной реализации это будет более сложная логика
        print("Запуск обучения для фолда...")
        
        # Сохранение результатов фолда (заглушка)
        fold_result = {
            'fold': fold_idx,
            'train_animals': train_animals,
            'val_animals': val_animals,
            'val_loss': np.random.rand(),  # Заглушка
            'val_acc': np.random.rand()    # Заглушка
        }
        fold_results.append(fold_result)
    
    # Вычисление средних результатов
    avg_val_loss = np.mean([r['val_loss'] for r in fold_results])
    avg_val_acc = np.mean([r['val_acc'] for r in fold_results])
    
    print(f"\n=== Результаты кросс-валидации ===")
    print(f"Средняя валидационная потеря: {avg_val_loss:.4f}")
    print(f"Средняя валидационная точность: {avg_val_acc:.4f}")
    
    # Сохранение результатов
    results_file = Path(config['experiment']['output_dir']) / 'cv_results.json'
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    cv_results = {
        'config': config,
        'n_splits': n_splits,
        'fold_results': fold_results,
        'avg_val_loss': float(avg_val_loss),
        'avg_val_acc': float(avg_val_acc)
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(cv_results, f, ensure_ascii=False, indent=2)
    
    print(f"Результаты кросс-валидации сохранены в: {results_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Кросс-валидация по животным')
    parser.add_argument('--config', type=str, required=True, 
                        help='Путь к файлу конфигурации')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Количество фолдов')
    
    args = parser.parse_args()
    
    run_cross_validation(args.config, args.n_splits)