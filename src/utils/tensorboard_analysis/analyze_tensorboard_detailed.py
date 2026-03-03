#!/usr/bin/env python3
"""
Скрипт для детального анализа данных из логов TensorBoard.
"""
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import yaml

def analyze_event_file(event_file_path):
    """
    Анализирует файл событий TensorBoard и выводит информацию о доступных данных.
    
    Args:
        event_file_path (str): Путь к файлу событий TensorBoard
    """
    print(f"Анализ файла: {event_file_path}")
    
    event_acc = EventAccumulator(event_file_path)
    event_acc.Reload()
    
    # Получаем информацию о доступных данных
    tags = event_acc.Tags()
    
    print("Доступные скалярные метрики:")
    for tag in tags['scalars']:
        print(f"  - {tag}")
    
    print("Доступные гистограммы:")
    for tag in tags.get('histograms', []):
        print(f"  - {tag}")
    
    print("Доступные изображения:")
    for tag in tags.get('images', []):
        print(f"  - {tag}")
    
    # Получаем скалярные данные
    print("\nСкалярные данные:")
    for tag in tags['scalars']:
        events = event_acc.Scalars(tag)
        if events:
            values = [event.value for event in events]
            print(f"  {tag}: {len(values)} значений, диапазон [{min(values):.6f}, {max(values):.6f}]")
    
    print("-" * 50)

def main():
    """
    Основная функция для анализа структуры TensorBoard логов.
    """
    # Определяем пути к логам экспериментов
    base_log_dir = "experiments/exp_001/logs/epilepsy_detector"
    
    experiment_runs = {
        "Run 7 (ConvBiGRUDetector)": "version_7",
        "Run 8 (TCNDetector)": "version_8",
        "Run 11 (UNet1DDetector)": "version_11"
    }
    
    for exp_name, version_dir in experiment_runs.items():
        log_dir = os.path.join(base_log_dir, version_dir)
        if os.path.exists(log_dir):
            print(f"\n=== {exp_name} ===")
            
            # Извлекаем гиперпараметры
            hparams_file = os.path.join(log_dir, 'hparams.yaml')
            if os.path.exists(hparams_file):
                with open(hparams_file, 'r') as f:
                    hparams = yaml.safe_load(f)
                print(f"Модель: {hparams.get('model_name', 'Unknown')}")
            
            # Анализируем все файлы событий в директории
            event_files = []
            for root, dirs, files in os.walk(log_dir):
                for file in files:
                    if file.startswith('events.out.tfevents'):
                        event_files.append(os.path.join(root, file))
            
            print(f"Найдено {len(event_files)} файлов событий")
            
            # Анализируем каждый файл событий
            for event_file in event_files:
                analyze_event_file(event_file)

if __name__ == "__main__":
    main()