#!/usr/bin/env python3
"""
Скрипт для комплексного извлечения данных из логов TensorBoard для анализа экспериментов.
"""
import os
import yaml

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_scalars_from_event_files(log_dir):
    """
    Извлекает скалярные метрики из всех файлов событий в директории.
    
    Args:
        log_dir (str): Директория с логами эксперимента
        
    Returns:
        dict: Словарь с метриками и их значениями
    """
    # Находим все файлы событий
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_files.append(os.path.join(root, file))
    
    # Собираем все метрики из всех файлов
    all_metrics = {}
    
    for event_file in event_files:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
        
        # Получаем скалярные метрики
        scalar_tags = event_acc.Tags()['scalars']
        
        for tag in scalar_tags:
            if tag in ['epoch', 'hp_metric', 'train/loss_step']:
                continue
                
            events = event_acc.Scalars(tag)
            if events:
                values = [event.value for event in events]
                if tag not in all_metrics:
                    all_metrics[tag] = []
                all_metrics[tag].extend(values)
    
    # Для каждой метрики сохраняем последние значение и лучшее значение
    metrics_data = {}
    for tag, values in all_metrics.items():
        if values:
            # Для метрик точности и F1 лучшее значение - максимальное
            if 'acc' in tag or 'f1' in tag or 'precision' in tag or 'recall' in tag:
                best_val = max(values)
            # Для метрик потерь лучшее значение - минимальное
            elif 'loss' in tag:
                best_val = min(values)
            # Для остальных - последнее значение
            else:
                best_val = values[-1]
            
            metrics_data[tag] = {
                'last': values[-1],
                'best': best_val,
                'values': values
            }
    
    return metrics_data

def extract_experiment_data(log_dir, experiment_name):
    """
    Извлекает данные для одного эксперимента.
    
    Args:
        log_dir (str): Директория с логами эксперимента
        experiment_name (str): Имя эксперимента
        
    Returns:
        dict: Данные эксперимента
    """
    experiment_data = {
        'name': experiment_name,
        'hparams': {},
        'metrics': {}
    }
    
    # Извлекаем гиперпараметры из файла hparams.yaml
    hparams_file = os.path.join(log_dir, 'hparams.yaml')
    if os.path.exists(hparams_file):
        with open(hparams_file, 'r') as f:
            experiment_data['hparams'] = yaml.safe_load(f)
    
    # Извлекаем метрики из файлов событий
    experiment_data['metrics'] = extract_scalars_from_event_files(log_dir)
    
    return experiment_data

def format_metric_value(value):
    """
    Форматирует значение метрики для отображения.
    """
    if value is None or (isinstance(value, str) and value == "N/A"):
        return "N/A"
    try:
        return f"{value:.4f}"
    except (ValueError, TypeError):
        return str(value)

def generate_experiment_summary(experiment_data):
    """
    Генерирует сводку по эксперименту.
    
    Args:
        experiment_data (dict): Данные эксперимента
        
    Returns:
        str: Сводка по эксперименту
    """
    summary = f"# Эксперимент: {experiment_data['name']}\n\n"
    
    # Информация о модели
    model_name = experiment_data['hparams'].get('model_name', 'Unknown')
    summary += f"## Модель: {model_name}\n\n"
    
    # Гиперпараметры
    summary += "## Гиперпараметры:\n\n"
    for key, value in experiment_data['hparams'].items():
        if isinstance(value, list):
            summary += f"- {key}: {value}\n"
        else:
            summary += f"- {key}: {value}\n"
    summary += "\n"
    
    # Метрики
    summary += "## Метрики:\n\n"
    
    # Обучающие метрики
    summary += "### Обучение:\n"
    train_metrics = {k: v for k, v in experiment_data['metrics'].items() if k.startswith('train/')}
    if train_metrics:
        for metric, values in train_metrics.items():
            metric_name = metric.replace('train/', '')
            last_val = format_metric_value(values.get('last', 'N/A'))
            best_val = format_metric_value(values.get('best', 'N/A'))
            summary += f"- {metric_name}: последнее = {last_val}, лучшее = {best_val}\n"
    else:
        summary += "- Нет данных\n"
    summary += "\n"
    
    # Валидационные метрики
    summary += "### Валидация:\n"
    val_metrics = {k: v for k, v in experiment_data['metrics'].items() if k.startswith('val/')}
    if val_metrics:
        for metric, values in val_metrics.items():
            metric_name = metric.replace('val/', '')
            last_val = format_metric_value(values.get('last', 'N/A'))
            best_val = format_metric_value(values.get('best', 'N/A'))
            summary += f"- {metric_name}: последнее = {last_val}, лучшее = {best_val}\n"
    else:
        summary += "- Нет данных\n"
    summary += "\n"
    
    # Тестовые метрики
    summary += "### Тестирование:\n"
    test_metrics = {k: v for k, v in experiment_data['metrics'].items() if k.startswith('test/')}
    if test_metrics:
        for metric, values in test_metrics.items():
            metric_name = metric.replace('test/', '')
            last_val = format_metric_value(values.get('last', 'N/A'))
            best_val = format_metric_value(values.get('best', 'N/A'))
            summary += f"- {metric_name}: последнее = {last_val}, лучшее = {best_val}\n"
    else:
        summary += "- Нет данных\n"
    summary += "\n"
    
    return summary

def generate_comparison_table(experiments_data):
    """
    Генерирует таблицу сравнения экспериментов.
    
    Args:
        experiments_data (list): Список данных экспериментов
        
    Returns:
        str: Таблица сравнения
    """
    # Создаем заголовок таблицы
    table = "| Модель | val_loss | val_acc | val_f1 | test_acc | test_f1 | test_precision | test_recall |\n"
    table += "|--------|----------|---------|--------|----------|---------|----------------|-------------|\n"
    
    # Заполняем строки таблицы
    for exp_data in experiments_data:
        model_name = exp_data['hparams'].get('model_name', 'Unknown')
        row = f"| {model_name} "
        
        # Извлекаем значения метрик
        metrics = exp_data['metrics']
        val_loss = metrics.get('val/loss', {}).get('best', 'N/A')
        val_acc = metrics.get('val/acc', {}).get('best', 'N/A')
        val_f1 = metrics.get('val/f1', {}).get('best', 'N/A')
        test_acc = metrics.get('test/acc', {}).get('last', 'N/A')
        test_f1 = metrics.get('test/f1', {}).get('last', 'N/A')
        test_precision = metrics.get('test/precision', {}).get('last', 'N/A')
        test_recall = metrics.get('test/recall', {}).get('last', 'N/A')
        
        # Форматируем значения
        val_loss_str = format_metric_value(val_loss)
        val_acc_str = format_metric_value(val_acc)
        val_f1_str = format_metric_value(val_f1)
        test_acc_str = format_metric_value(test_acc)
        test_f1_str = format_metric_value(test_f1)
        test_precision_str = format_metric_value(test_precision)
        test_recall_str = format_metric_value(test_recall)
        
        row += f"| {val_loss_str} | {val_acc_str} | {val_f1_str} | {test_acc_str} | {test_f1_str} | {test_precision_str} | {test_recall_str} |"
        table += row + "\n"
    
    return table

def main():
    """
    Основная функция для извлечения и анализа данных.
    """
    # Определяем пути к логам экспериментов
    base_log_dir = "experiments/exp_001/logs/epilepsy_detector"
    
    experiment_runs = {
        "Run 7 (ConvBiGRUDetector)": "version_7",
        "Run 8 (TCNDetector)": "version_8",
        "Run 11 (UNet1DDetector)": "version_11"
    }
    
    # Извлекаем данные для каждого эксперимента
    experiments_data = []
    experiment_summaries = []
    
    for exp_name, version_dir in experiment_runs.items():
        log_dir = os.path.join(base_log_dir, version_dir)
        if os.path.exists(log_dir):
            print(f"Обработка {exp_name}...")
            exp_data = extract_experiment_data(log_dir, exp_name)
            experiments_data.append(exp_data)
            summary = generate_experiment_summary(exp_data)
            experiment_summaries.append(summary)
        else:
            print(f"Директория {log_dir} не найдена")
    
    # Генерируем сводную таблицу
    comparison_table = generate_comparison_table(experiments_data)
    
    # Сохраняем отчеты
    # Индивидуальные отчеты
    for i, summary in enumerate(experiment_summaries):
        model_name = experiments_data[i]['hparams'].get('model_name', 'Unknown')
        filename = f"experiment_{model_name}_summary.md"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Создан отчет: {filename}")
    
    # Сводный отчет
    with open("experiments_comprehensive_comparison.md", 'w', encoding='utf-8') as f:
        f.write("# Комплексный сравнительный анализ экспериментов\n\n")
        f.write("## Сводная таблица\n\n")
        f.write(comparison_table)
        f.write("\n\n## Детализированные отчеты по каждому эксперименту\n\n")
        for summary in experiment_summaries:
            f.write(summary)
            f.write("\n\n---\n\n")
    
    print("Создан сводный отчет: experiments_comprehensive_comparison.md")
    print("Анализ завершен!")

if __name__ == "__main__":
    main()