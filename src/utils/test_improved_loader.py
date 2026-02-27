#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тестовый скрипт для проверки функциональности улучшенного загрузчика данных
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
import sys

def load_single_processed_recording(animal_id, session_id, data_dir="test_output_small"):
    """
    Загрузка одной записи из предобработанных данных
    
    Параметры:
    animal_id (str): идентификатор животного
    session_id (str): идентификатор сессии
    data_dir (str): путь к директории с обработанными данными
    
    Возвращает:
    tuple: (сигналы, сегменты, метаданные, маска приступов)
    """
    session_path = Path(data_dir) / animal_id / session_id
    
    if not session_path.exists():
        raise FileNotFoundError(f"Сессия {animal_id}/{session_id} не найдена")
    
    # Загрузка обработанных сигналов
    signals_file = session_path / "processed_signals.npy"
    if not signals_file.exists():
        raise FileNotFoundError(f"Файл сигналов не найден: {signals_file}")
    signals = np.load(signals_file)
    
    # Загрузка информации о сегментах
    segments_file = session_path / "segments_info.csv"
    if not segments_file.exists():
        raise FileNotFoundError(f"Файл сегментов не найден: {segments_file}")
    segments_df = pd.read_csv(segments_file)
    
    # Загрузка маски приступов
    seizure_mask_file = session_path / "seizure_mask.npy"
    if not seizure_mask_file.exists():
        raise FileNotFoundError(f"Файл маски приступов не найден: {seizure_mask_file}")
    seizure_mask = np.load(seizure_mask_file)
    
    # Загрузка метаданных
    metadata_file = session_path / "conversion_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    else:
        metadata = {}
    
    return signals, segments_df, metadata, seizure_mask

def main():
    """Основная функция для тестирования"""
    print("Тестирование улучшенного загрузчика данных...")
    
    try:
        # Загрузка одной записи
        animal_id = "Dex1x2NE"
        session_id = "BL_24Jan"
        
        print(f"Загрузка данных для животного {animal_id}, сессия {session_id}")
        signals, segments_df, metadata, seizure_mask = load_single_processed_recording(animal_id, session_id)
        
        print(f"Формат сигналов: {signals.shape}")
        print(f"Количество сегментов: {len(segments_df)}")
        print(f"Формат маски приступов: {seizure_mask.shape}")
        print(f"Частота дискретизации: {metadata.get('sampling_freq', 'Не указана')} Гц")
        print(f"Каналы: {metadata.get('channel_names', 'Не указаны')}")
        
        # Проверка сегментов
        seizure_segments = segments_df[segments_df['segment_type'] == 'seizure']
        print(f"Количество сегментов с приступами: {len(seizure_segments)}")
        
        print("\nТест успешно пройден!")
        return True
        
    except Exception as e:
        print(f"Ошибка при тестировании: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)