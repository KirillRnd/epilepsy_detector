#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для запуска кросс-валидации модели детектирования эпилепсии
"""

import argparse
import sys
import os

# Добавляем путь к модулям проекта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.cross_validation import run_cross_validation

def main():
    """
    Основная функция для запуска кросс-валидации
    """
    parser = argparse.ArgumentParser(description='Кросс-валидация модели детектирования эпилепсии')
    parser.add_argument('--config', type=str, required=True, 
                        help='Путь к файлу конфигурации')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Количество фолдов (по умолчанию: 5)')
    
    args = parser.parse_args()
    
    print("Запуск кросс-валидации...")
    print(f"Конфигурационный файл: {args.config}")
    print(f"Количество фолдов: {args.n_splits}")
    
    # Запуск кросс-валидации
    run_cross_validation(args.config, args.n_splits)
    
    print("Кросс-валидация завершена!")

if __name__ == "__main__":
    main()