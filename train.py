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
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import shutil
from datetime import datetime

# Добавляем путь к модулям проекта
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_loading.epilepsy_datamodule import EpilepsyDataModule
from src.modeling.lightning_epilepsy_detector import EpilepsyDetector_v2
from src.preprocessing.lightning_class_balancer import compute_class_weights


class FinalMetricsMarkdownCallback(Callback):
    """Writes final validation and test metrics for the selected checkpoint."""

    def __init__(self, filename: str = "final_metrics.md"):
        super().__init__()
        self.filename = filename

    @staticmethod
    def _first_result(results):
        if isinstance(results, list):
            return results[0] if results else {}
        return results or {}

    @staticmethod
    def _format_value(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.numel() == 1:
                value = value.item()
            else:
                value = value.tolist()
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, float):
            return f"{value:.10g}"
        return str(value)

    def _metrics_table(self, title: str, metrics: dict) -> list:
        lines = [f"## {title}", "", "| Metric | Value |", "| --- | ---: |"]
        if not metrics:
            lines.append("| _No metrics returned_ |  |")
        else:
            for key in sorted(metrics):
                lines.append(f"| `{key}` | {self._format_value(metrics[key])} |")
        lines.append("")
        return lines

    def write_report(
        self,
        experiment_dir: str,
        val_results,
        test_results,
        checkpoint_callback: ModelCheckpoint,
        config: dict,
        readable_checkpoint_path: str = "",
    ) -> str:
        os.makedirs(experiment_dir, exist_ok=True)
        report_path = os.path.join(experiment_dir, self.filename)

        best_score = checkpoint_callback.best_model_score
        if isinstance(best_score, torch.Tensor):
            best_score = best_score.detach().cpu().item()

        lines = [
            "# Final metrics",
            "",
            f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"- Monitor: `{checkpoint_callback.monitor}`",
            f"- Best score: {self._format_value(best_score) if best_score is not None else 'n/a'}",
            f"- Best checkpoint: `{checkpoint_callback.best_model_path or 'n/a'}`",
        ]
        if readable_checkpoint_path:
            lines.append(f"- Readable checkpoint copy: `{readable_checkpoint_path}`")
        lines.append("")
        lines.extend(self._metrics_table("Validation metrics", self._first_result(val_results)))
        lines.extend(self._metrics_table("Test metrics", self._first_result(test_results)))
        lines.extend([
            "## Config",
            "",
            "```yaml",
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True).strip(),
            "```",
            "",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        return report_path


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
        seed=config['experiment']['seed'],
        cache_mode=config['data'].get('cache_mode', 'mmap'),
        max_open_files=config['data'].get('max_open_files', 16),
        num_workers=config['data'].get('num_workers', 4),
        pin_memory=config['data'].get('pin_memory', True),
        persistent_workers=config['data'].get('persistent_workers', True),
        prefetch_factor=config['data'].get('prefetch_factor', 2),
        train_shuffle_mode=config['data'].get('train_shuffle_mode', 'block'),
        block_shuffle_size=config['data'].get('block_shuffle_size', 4096),
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
    experiment_dir = logger.log_dir
    checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Создание callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='epilepsy-detector-{epoch:02d}-val_f1={val/f1:.5f}',
        save_top_k=3,
        monitor='val/f1',
        mode='max',
        save_last=True,
        auto_insert_metric_name=False,
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val/f1',
        patience=config['training']['patience'],
        verbose=True,
        mode='max'
    )
    final_metrics_callback = FinalMetricsMarkdownCallback()
    
    # Создание тренера
    trainer = pl.Trainer(
        max_epochs=config['training']['num_epochs'],
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, final_metrics_callback],
        accelerator='gpu' if torch.cuda.is_available() and config['experiment']['device'] == 'cuda' else 'cpu',
        devices=1,
        log_every_n_steps=config['training'].get('log_every_n_steps', 10),
        deterministic=True,
        limit_train_batches=config['training'].get('limit_train_batches', 1.0),
        limit_val_batches=config['training'].get('limit_val_batches', 1.0),
        limit_test_batches=config['training'].get('limit_test_batches', 1.0),
        num_sanity_val_steps=config['training'].get('num_sanity_val_steps', 2),
    )
    
    # Обучение модели
    print("Начало обучения...")
    trainer.fit(model, datamodule=data_module)
    readable_path = ""

    # Сохранение лучшего чекпоинта в читаемом виде с указанием модели
    if checkpoint_callback.best_model_path:
        best_path = checkpoint_callback.best_model_path
        best_f1 = float(checkpoint_callback.best_model_score)
        model_name = config['model']['model_name']
        # Создать читаемое имя
        readable_name = f"{model_name}-best-val_f1={best_f1:.5f}.ckpt"
        readable_path = os.path.join(checkpoint_dir, readable_name)
        # Копировать файл
        shutil.copy2(best_path, readable_path)
        print(f"Создан читаемый чекпоинт: {readable_path}")
    else:
        print("Предупреждение: лучший чекпоинт не найден.")

    # Валидация и тестирование на лучшем чекпоинте
    if checkpoint_callback.best_model_path:
        best_path = checkpoint_callback.best_model_path
        print(f"Начало валидации на лучшем чекпоинте: {best_path}")
        val_results = trainer.validate(model, datamodule=data_module, ckpt_path=best_path)
        
        print("Начало тестирования на лучшем чекпоинте...")
        data_module.setup(stage='test')
        test_results = trainer.test(model, datamodule=data_module, ckpt_path=best_path)
    else:
        print("Предупреждение: лучший чекпоинт не найден, используется последняя модель.")
        print("Начало валидации...")
        val_results = trainer.validate(model, datamodule=data_module)
        
        print("Начало тестирования на тестовых данных...")
        data_module.setup(stage='test')
        test_results = trainer.test(model, datamodule=data_module)

    report_path = final_metrics_callback.write_report(
        experiment_dir=experiment_dir,
        val_results=val_results,
        test_results=test_results,
        checkpoint_callback=checkpoint_callback,
        config=config,
        readable_checkpoint_path=readable_path,
    )
    print(f"Final metrics saved: {report_path}")
    
    print("Обучение завершено!")


if __name__ == "__main__":
    main()
