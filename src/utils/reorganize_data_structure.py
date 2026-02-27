#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для реорганизации структуры данных в проекте детектора эпилепсии у крыс.

Реорганизует файлы в data/raw, создавая правильную структуру с папками сессий
для каждого животного.
"""

import os
import re
import shutil
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reorganization.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_session_info(filename):
    """
    Извлекает информацию о сессии из имени файла
    
    Параметры:
    filename (str): имя файла
    
    Возвращает:
    dict: словарь с информацией о сессии
    """
    logger.debug(f"Извлечение информации из файла: {filename}")
    
    # Извлечение даты
    date_patterns = [
        r'(\d{1,2}[A-Z][a-z]+)',  # 10May, 21June
        r'(\d{1,2}[A-Z][a-z]{2,})', # 10January
        r'(\d{1,2}[A-Za-z]{3,})' # различные форматы дат
    ]
    
    date_str = None
    for pattern in date_patterns:
        match = re.search(pattern, filename)
        if match:
            date_str = match.group(1)
            break
    
    # Извлечение условия
    conditions = ['BL', 'H2O', 'estrus', 'diestr', 'proestr', 'metestr']
    condition = 'unknown'
    for cond in conditions:
        if cond.lower() in filename.lower():
            condition = cond
            break
    
    # Формирование session_id
    if date_str and condition != 'unknown':
        session_id = f"{condition}_{date_str}"
    elif date_str:
        session_id = f"session_{date_str}"
    else:
        # Если не удалось извлечь дату, используем часть имени файла
        parts = filename.split('_')
        if len(parts) > 1:
            session_id = f"{parts[0]}_{parts[1]}"
        else:
            session_id = "unknown_session"
    
    logger.debug(f"Извлечена сессия: {session_id}, дата: {date_str}, условие: {condition}")
    
    return {
        'session_id': session_id,
        'date': date_str,
        'condition': condition
    }

def reorganize_animal_data(animal_dir):
    """
    Реорганизует данные для одного животного
    
    Параметры:
    animal_dir (str): путь к папке животного
    """
    logger.info(f"Обработка данных для животного: {animal_dir}")
    
    # Получение списка всех файлов
    all_files = []
    for root, dirs, files in os.walk(animal_dir):
        for file in files:
            if file.endswith(('.edf', '.adicht', '.txt')):
                all_files.append(os.path.join(root, file))
    
    logger.info(f"Найдено файлов: {len(all_files)}")
    
    # Группировка файлов по сессиям
    sessions = {}
    for file_path in all_files:
        filename = os.path.basename(file_path)
        session_info = extract_session_info(filename)
        session_id = session_info['session_id']
        
        if session_id not in sessions:
            sessions[session_id] = {
                'info': session_info,
                'files': []
            }
        sessions[session_id]['files'].append(file_path)
    
    logger.info(f"Найдено сессий: {len(sessions)}")
    
    # Создание папок сессий и перемещение файлов
    for session_id, session_data in sessions.items():
        session_dir = os.path.join(animal_dir, session_id)
        os.makedirs(session_dir, exist_ok=True)
        logger.info(f"Создана папка сессии: {session_dir}")
        
        for file_path in session_data['files']:
            filename = os.path.basename(file_path)
            new_path = os.path.join(session_dir, filename)
            
            # Обработка конфликтов имен
            counter = 1
            base_name, ext = os.path.splitext(filename)
            while os.path.exists(new_path):
                new_name = f"{base_name}_{counter}{ext}"
                new_path = os.path.join(session_dir, new_name)
                counter += 1
            
            try:
                shutil.move(file_path, new_path)
                logger.debug(f"Файл перемещен: {file_path} -> {new_path}")
            except Exception as e:
                logger.error(f"Ошибка при перемещении файла {file_path}: {e}")
    
    # Создание метаданных
    create_animal_metadata(animal_dir, sessions)
    
    logger.info(f"Обработка животного {animal_dir} завершена")

def create_animal_metadata(animal_dir, sessions):
    """
    Создает файл метаданных для животного
    
    Параметры:
    animal_dir (str): путь к папке животного
    sessions (dict): словарь с информацией о сессиях
    """
    logger.info(f"Создание метаданных для: {animal_dir}")
    
    metadata_rows = []
    for session_id, session_data in sessions.items():
        for file_path in session_data['files']:
            filename = os.path.basename(file_path)
            metadata_rows.append({
                'session_id': session_id,
                'file_name': filename,
                'condition': session_data['info']['condition'],
                'date': session_data['info']['date']
            })
    
    if metadata_rows:
        metadata_df = pd.DataFrame(metadata_rows)
        metadata_path = os.path.join(animal_dir, 'metadata.csv')
        metadata_df.to_csv(metadata_path, index=False, encoding='utf-8')
        logger.info(f"Метаданные сохранены: {metadata_path}")

def main():
    """
    Основная функция реорганизации данных
    """
    logger.info("Начало реорганизации структуры данных")
    
    raw_data_dir = 'data/raw'
    
    # Проверка существования директории
    if not os.path.exists(raw_data_dir):
        logger.error(f"Директория {raw_data_dir} не существует")
        return
    
    # Получение списка животных
    try:
        animal_dirs = [d for d in os.listdir(raw_data_dir) 
                       if os.path.isdir(os.path.join(raw_data_dir, d))]
        logger.info(f"Найдено животных: {len(animal_dirs)}")
    except Exception as e:
        logger.error(f"Ошибка при получении списка животных: {e}")
        return
    
    # Реорганизация данных для каждого животного
    for animal_id in animal_dirs:
        animal_dir = os.path.join(raw_data_dir, animal_id)
        try:
            reorganize_animal_data(animal_dir)
        except Exception as e:
            logger.error(f"Ошибка при обработке {animal_id}: {e}")
    
    logger.info("Реорганизация завершена!")

if __name__ == "__main__":
    main()