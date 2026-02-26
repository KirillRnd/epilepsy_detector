# План экспериментов по обучению и валидации модели детектора эпилепсии

## Общая стратегия экспериментов

План экспериментов включает следующие этапы:
1. Базовая настройка и подготовка данных
2. Обучение базовой модели
3. Оптимизация гиперпараметров
4. Сравнение архитектур
5. Валидация на отложенной выборке
6. Анализ результатов и выводы

## 1. Базовая настройка и подготовка данных

### Подготовка датасета
```python
def prepare_baseline_dataset():
    """
    Подготовка базового датасета для экспериментов
    """
    experiment_config = {
        # Параметры данных
        'data_dir': 'data/processed',
        'window_length': 5.0,  # секунды
        'window_overlap': 0.5,  # 50% перекрытие
        'target_sampling_rate': 512.0,
        'channels': ['EEG1', 'EEG2', 'EEG3', 'EEG4'],  # или все доступные каналы
        
        # Балансировка классов
        'seizure_ratio': 0.25,
        'augmentation_factor': 2,
        
        # Разделение данных
        'train_ratio': 0.7,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
        
        # Кросс-валидация
        'cv_folds': 5,
        'cv_by_animal': True  # Разделение по животным
    }
    
    return experiment_config
```

### Создание загрузчиков данных
```python
def create_data_loaders(config):
    """
    Создание загрузчиков данных для экспериментов
    
    Параметры:
    config (dict): конфигурация эксперимента
    
    Возвращает:
    tuple: (train_loader, val_loader, test_loader)
    """
    from torch.utils.data import DataLoader, Subset
    from sklearn.model_selection import train_test_split
    
    # Загрузка предобработанных данных
    dataset = EEGDataset(
        data_dir=config['data_dir'],
        window_length=config['window_length'],
        target_sampling_rate=config['target_sampling_rate'],
        channels=config['channels']
    )
    
    # Разделение на train/val/test
    indices = list(range(len(dataset)))
    train_indices, temp_indices = train_test_split(
        indices, 
        train_size=config['train_ratio'],
        stratify=[dataset[i][1] for i in indices],  # По меткам классов
        random_state=42
    )
    
    val_indices, test_indices = train_test_split(
        temp_indices,
        train_size=config['val_ratio']/(config['val_ratio']+config['test_ratio']),
        stratify=[dataset[i][1] for i in temp_indices],
        random_state=42
    )
    
    # Создание подмножеств
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Создание загрузчиков
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader, test_loader
```

## 2. Обучение базовой модели

### Эксперимент 1: Базовая 1D-CNN
```python
def experiment_001_baseline_cnn():
    """
    Эксперимент 1: Базовая 1D-CNN модель
    """
    experiment_config = {
        'experiment_id': '001_baseline_cnn',
        'model_type': 'SimpleEEGDetector',
        'model_params': {
            'input_channels': 4,
            'window_length': 2560,  # 5 сек при 512 Гц
            'num_classes': 2,
            'dropout_rate': 0.5
        },
        'training_params': {
            'num_epochs': 100,
            'learning_rate': 0.001,
            'batch_size': 64,
            'weight_decay': 1e-4
        },
        'data_config': prepare_baseline_dataset()
    }
    
    # Создание модели
    model = SimpleEEGDetector(**experiment_config['model_params'])
    
    # Подготовка данных
    train_loader, val_loader, test_loader = create_data_loaders(
        experiment_config['data_config']
    )
    
    # Обучение модели
    history = train_model(
        model, train_loader, val_loader,
        num_epochs=experiment_config['training_params']['num_epochs'],
        learning_rate=experiment_config['training_params']['learning_rate'],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Оценка модели
    metrics = evaluate_model(model, test_loader)
    
    # Сохранение результатов
    save_experiment_results(
        experiment_config['experiment_id'],
        model, history, metrics, experiment_config
    )
    
    return metrics
```

## 3. Оптимизация гиперпараметров

### Эксперимент 2: Поиск оптимальной архитектуры
```python
def experiment_002_hyperparameter_search():
    """
    Эксперимент 2: Поиск оптимальных гиперпараметров
    """
    # Диапазоны для поиска
    param_grid = {
        'learning_rate': [0.0001, 0.001, 0.01],
        'batch_size': [32, 64, 128],
        'dropout_rate': [0.3, 0.5, 0.7],
        'num_filters': [32, 64, 128]
    }
    
    best_metrics = None
    best_params = None
    best_score = 0.0
    
    # Поиск по сетке
    for lr in param_grid['learning_rate']:
        for bs in param_grid['batch_size']:
            for dr in param_grid['dropout_rate']:
                for nf in param_grid['num_filters']:
                    # Конфигурация эксперимента
                    config = {
                        'model_params': {
                            'input_channels': 4,
                            'window_length': 2560,
                            'num_classes': 2,
                            'dropout_rate': dr,
                            'num_filters': nf
                        },
                        'training_params': {
                            'num_epochs': 50,  # Уменьшено для поиска
                            'learning_rate': lr,
                            'batch_size': bs
                        }
                    }
                    
                    # Создание и обучение модели
                    model = SimpleEEGDetector(**config['model_params'])
                    train_loader, val_loader, _ = create_data_loaders(config)
                    
                    history = train_model(
                        model, train_loader, val_loader,
                        num_epochs=config['training_params']['num_epochs'],
                        learning_rate=config['training_params']['learning_rate']
                    )
                    
                    # Оценка на валидационной выборке
                    _, val_acc, _, _ = validate_epoch(model, val_loader)
                    
                    # Обновление лучшего результата
                    if val_acc > best_score:
                        best_score = val_acc
                        best_params = config
                        best_metrics = {
                            'val_accuracy': val_acc,
                            'history': history
                        }
    
    return best_params, best_metrics
```

### Эксперимент 3: Сравнение функций активации
```python
def experiment_003_activation_comparison():
    """
    Эксперимент 3: Сравнение различных функций активации
    """
    activations = ['relu', 'leaky_relu', 'elu', 'swish']
    results = {}
    
    for activation in activations:
        # Модифицированная модель с разными активациями
        model = EEGDetectorWithActivation(
            input_channels=4,
            window_length=2560,
            num_classes=2,
            activation=activation
        )
        
        # Обучение модели
        train_loader, val_loader, test_loader = create_data_loaders()
        history = train_model(model, train_loader, val_loader, num_epochs=50)
        
        # Оценка
        metrics = evaluate_model(model, test_loader)
        results[activation] = metrics
        
        print(f"Activation {activation}: Accuracy = {metrics['accuracy']:.4f}")
    
    return results
```

## 4. Сравнение архитектур

### Эксперимент 4: Базовая CNN vs Улучшенная CNN
```python
def experiment_004_architecture_comparison():
    """
    Эксперимент 4: Сравнение базовой и улучшенной архитектур
    """
    architectures = {
        'simple_cnn': SimpleEEGDetector,
        'improved_cnn': ImprovedEEGDetector
    }
    
    results = {}
    
    for name, model_class in architectures.items():
        print(f"Training {name}...")
        
        # Создание модели
        if name == 'simple_cnn':
            model = model_class(input_channels=4, window_length=2560, num_classes=2)
        else:
            model = model_class(input_channels=4, window_length=2560, num_classes=2)
        
        # Подготовка данных
        train_loader, val_loader, test_loader = create_data_loaders()
        
        # Обучение
        history = train_model(model, train_loader, val_loader, num_epochs=100)
        
        # Оценка
        metrics = evaluate_model(model, test_loader)
        results[name] = {
            'metrics': metrics,
            'history': history
        }
        
        print(f"{name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_score']:.4f}")
    
    return results
```

### Эксперимент 5: Влияние количества каналов
```python
def experiment_005_channel_influence():
    """
    Эксперимент 5: Влияние количества каналов на качество детектирования
    """
    channel_configs = [
        ['EEG1'],                    # 1 канал
        ['EEG1', 'EEG2'],           # 2 канала
        ['EEG1', 'EEG2', 'EEG3'],   # 3 канала
        ['EEG1', 'EEG2', 'EEG3', 'EEG4']  # 4 канала
    ]
    
    results = {}
    
    for channels in channel_configs:
        print(f"Testing with channels: {channels}")
        
        # Модифицированная подготовка данных
        train_loader, val_loader, test_loader = create_data_loaders_with_channels(channels)
        
        # Создание модели с соответствующим количеством каналов
        model = SimpleEEGDetector(
            input_channels=len(channels),
            window_length=2560,
            num_classes=2
        )
        
        # Обучение
        history = train_model(model, train_loader, val_loader, num_epochs=50)
        
        # Оценка
        metrics = evaluate_model(model, test_loader)
        results[len(channels)] = metrics
        
        print(f"Channels {len(channels)}: Accuracy = {metrics['accuracy']:.4f}")
    
    return results
```

## 5. Валидация на отложенной выборке

### Эксперимент 6: Кросс-валидация по животным
```python
def experiment_006_cross_validation():
    """
    Эксперимент 6: Кросс-валидация с разделением по животным
    """
    from sklearn.model_selection import KFold
    
    # Получение списка уникальных животных
    animals = get_unique_animals_from_dataset()
    
    # K-fold кросс-валидация по животным
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    
    cv_results = []
    
    for fold, (train_animals, test_animals) in enumerate(kfold.split(animals)):
        print(f"Fold {fold + 1}/5")
        
        # Создание датасетов для текущего fold
        train_dataset = create_dataset_for_animals(train_animals)
        test_dataset = create_dataset_for_animals(test_animals)
        
        # Создание загрузчиков
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Создание и обучение модели
        model = ImprovedEEGDetector(input_channels=4, window_length=2560, num_classes=2)
        history = train_model(model, train_loader, test_loader, num_epochs=100)
        
        # Оценка
        metrics = evaluate_model(model, test_loader)
        cv_results.append(metrics)
        
        print(f"Fold {fold + 1} - Accuracy: {metrics['accuracy']:.4f}")
    
    # Вычисление средних результатов
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in cv_results]),
        'f1_score': np.mean([m['f1_score'] for m in cv_results]),
        'sensitivity': np.mean([m['sensitivity'] for m in cv_results]),
        'specificity': np.mean([m['specificity'] for m in cv_results])
    }
    
    return cv_results, avg_metrics
```

## 6. Анализ результатов и выводы

### Сводный отчет по экспериментам
```python
def generate_experiment_report(all_results):
    """
    Генерация сводного отчета по всем экспериментам
    
    Параметры:
    all_results (dict): результаты всех экспериментов
    """
    report = {
        'summary': {},
        'best_model': None,
        'recommendations': []
    }
    
    # Анализ результатов каждого эксперимента
    for exp_id, results in all_results.items():
        report['summary'][exp_id] = {
            'accuracy': results['metrics']['accuracy'],
            'f1_score': results['metrics']['f1_score'],
            'sensitivity': results['metrics']['sensitivity'],
            'specificity': results['metrics']['specificity']
        }
    
    # Определение лучшей модели
    best_exp = max(all_results.keys(), 
                   key=lambda x: all_results[x]['metrics']['f1_score'])
    report['best_model'] = best_exp
    
    # Рекомендации на основе результатов
    if '004_architecture_comparison' in all_results:
        arch_results = all_results['004_architecture_comparison']
        if arch_results['improved_cnn']['metrics']['f1_score'] > \
           arch_results['simple_cnn']['metrics']['f1_score']:
            report['recommendations'].append(
                "Улучшенная архитектура с остаточными блоками показывает лучшие результаты"
            )
    
    if '005_channel_influence' in all_results:
        channel_results = all_results['005_channel_influence']
        best_channels = max(channel_results.keys(), 
                         key=lambda x: channel_results[x]['f1_score'])
        report['recommendations'].append(
            f"Оптимальное количество каналов: {best_channels}"
        )
    
    return report
```

## План проведения экспериментов

### Этап 1: Подготовка (1-2 дня)
1. Подготовка предобработанных данных
2. Создание базовых загрузчиков данных
3. Настройка инфраструктуры для экспериментов

### Этап 2: Базовые эксперименты (3-5 дней)
1. Эксперимент 1: Базовая 1D-CNN
2. Эксперимент 4: Сравнение архитектур
3. Эксперимент 5: Влияние количества каналов

### Этап 3: Оптимизация (5-7 дней)
1. Эксперимент 2: Поиск гиперпараметров
2. Эксперимент 3: Сравнение функций активации
3. Доработка лучшей модели

### Этап 4: Валидация (2-3 дня)
1. Эксперимент 6: Кросс-валидация по животным
2. Финальная оценка на тестовой выборке
3. Анализ ошибок и чувствительности модели

### Этап 5: Финальный анализ (1-2 дня)
1. Генерация сводного отчета
2. Подготовка рекомендаций по улучшению
3. Документирование результатов

## Метрики для оценки экспериментов

### Основные метрики:
1. **Accuracy** - общая точность классификации
2. **F1-Score** - гармоническое среднее precision и recall
3. **Sensitivity (Recall)** - доля правильно обнаруженных приступов
4. **Specificity** - доля правильно классифицированных нормальных участков
5. **AUC-ROC** - площадь под ROC кривой

### Дополнительные метрики:
1. **Precision** - точность положительных предсказаний
2. **False Positive Rate** - частота ложных срабатываний
3. **False Negative Rate** - частота пропущенных приступов
4. **Training Time** - время обучения модели
5. **Inference Time** - время предсказания для одного окна

## Рекомендации по проведению экспериментов

1. **Воспроизводимость**: Фиксировать random seeds для всех экспериментов
2. **Логирование**: Подробно логировать параметры и результаты каждого эксперимента
3. **Версионирование**: Использовать систему контроля версий для кода экспериментов
4. **Резервное копирование**: Регулярно сохранять промежуточные результаты
5. **Параллелизм**: Использовать параллельное выполнение экспериментов при возможности

## Возможные улучшения

1. **Автоматизация**: Создать pipeline для автоматического запуска серии экспериментов
2. **Визуализация**: Разработать dashboard для мониторинга результатов экспериментов
3. **Early Stopping**: Реализовать механизм ранней остановки для экономии времени
4. **Model Checkpointing**: Сохранять промежуточные состояния моделей
5. **Анализ ошибок**: Проводить детальный анализ типов ошибок моделей