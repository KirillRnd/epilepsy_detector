#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обучения модели детектирования эпилепсии
"""

import os
import argparse
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import shutil

# Добавляем путь к модулям проекта
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loading.epilepsy_datamodule import EpilepsyDataModule
from src.modeling.lightning_epilepsy_detector import EpilepsyDetector_v2
from src.preprocessing.lightning_class_balancer import compute_class_weights


def load_config(config_path: str) -> dict:
    """
    Загрузка конфигурации из YAML файла
    
    Параметры:
    config_path (str): путь к файлу конфигурации
    
    Возвращает:
    dict: конфигурация
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """
    Основная функция для запуска обучения
    """
    parser = argparse.ArgumentParser(description='Обучение модели детектирования эпилепсии')
    parser.add_argument('--config', type=str, required=True, 
                        help='Путь к файлу конфигурации')
    parser.add_argument('--test', action='store_true',
                        help='Запустить тестирование обученной модели')
    
    args = parser.parse_args()
    
    # Загрузка конфигурации
    config = load_config(args.config)
    
    # Установка seed для воспроизводимости
    pl.seed_everything(config['experiment']['seed'])
    
    # Определение устройства
    device = torch.device(config['experiment']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Используемое устройство: {device}")
    
    # Создание директорий для результатов
    os.makedirs(config['experiment']['output_dir'], exist_ok=True)
    os.makedirs(config['experiment']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['experiment']['log_dir'], exist_ok=True)
    
    # Создание DataModule
    data_module = EpilepsyDataModule(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        window_length=config['data']['window_length'],
        overlap=config['data']['overlap'],
        train_animal_ratio=config['data']['train_animal_ratio'],
        val_animal_ratio=config['data']['val_animal_ratio'],
        train_animals=config['data'].get('train_animals'),
        val_animals=config['data'].get('val_animals'),
        test_animals=config['data'].get('test_animals'),
        seed=config['experiment']['seed']
    )
    
    # Подготовка данных
    data_module.prepare_data()
    data_module.setup(stage='fit')
    
    # Создание модели
    model = EpilepsyDetector_v2(
        input_channels=config['model']['input_channels'],
        window_length=config['model']['window_length'],
        num_classes=config['model']['num_classes'],
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        model_name=config['model']['model_name'],
        class_weights=config['model']['class_weights']
    )
    
    # Создание логгера
    logger = TensorBoardLogger(
        config['experiment']['log_dir'], 
        name="epilepsy_detector"
    )
    
    # Создание callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config['experiment']['checkpoint_dir'],
        filename='epilepsy-detector-{epoch:02d}-{val/loss:.5f}',
        save_top_k=3,
        monitor='val/loss',
        mode='min',
        save_last=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/loss',
        patience=config['training']['patience'],
        verbose=True,
        mode='min'
    )
    
    # Создание тренера
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator='gpu' if torch.cuda.is_available() and config['experiment']['device'] == 'cuda' else 'cpu',
        devices=1,
        log_every_n_steps=10,
        deterministic=True
    )
    
    # Обучение модели
    print("Начало обучения...")
    trainer.fit(model, datamodule=data_module)

    # Сохранение лучшего чекпоинта в читаемом виде с указанием модели
    if checkpoint_callback.best_model_path:
        best_path = checkpoint_callback.best_model_path
        best_loss = float(checkpoint_callback.best_model_score)
        model_name = config['model']['model_name']
        # Создать читаемое имя
        readable_name = f"{model_name}-best-val_loss={best_loss:.5f}.ckpt"
        readable_path = os.path.join(config['experiment']['checkpoint_dir'], readable_name)
        # Копировать файл
        shutil.copy2(best_path, readable_path)
        print(f"Создан читаемый чекпоинт: {readable_path}")
    else:
        print("Предупреждение: лучший чекпоинт не найден.")

    # Валидация и тестирование на лучшем чекпоинте
    if checkpoint_callback.best_model_path:
        best_path = checkpoint_callback.best_model_path
        print(f"Начало валидации на лучшем чекпоинте: {best_path}")
        trainer.validate(model, datamodule=data_module, ckpt_path=best_path)
        
        print("Начало тестирования на лучшем чекпоинте...")
        data_module.setup(stage='test')
        trainer.test(model, datamodule=data_module, ckpt_path=best_path)
    else:
        print("Предупреждение: лучший чекпоинт не найден, используется последняя модель.")
        print("Начало валидации...")
        trainer.validate(model, datamodule=data_module)
        
        print("Начало тестирования на тестовых данных...")
        data_module.setup(stage='test')
        trainer.test(model, datamodule=data_module)
    
    print("Обучение завершено!")


if __name__ == "__main__":
    main()