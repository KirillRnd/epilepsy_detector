# План реализации этапа 2 в формате PyTorch Lightning

## 1. PyTorch Lightning DataModule для загрузки предобработанных данных

### 1.1. EpilepsyDataset
- Создать класс `EpilepsyDataset`, наследуемый от `torch.utils.data.Dataset`
- Реализовать методы `__init__`, `__len__` и `__getitem__`
- Загружать данные из .npy файлов (`processed_signals.npy` и `seizure_mask.npy`)
- Создавать скользящие окна фиксированной длины (например, 5 секунд при 400 Гц = 2000 отсчетов)
- Возвращать кортеж: (сигнал, метка)

### 1.2. EpilepsyDataModule
- Создать класс `EpilepsyDataModule`, наследуемый от `pytorch_lightning.LightningDataModule`
- Реализовать методы:
  - `__init__`: инициализация с параметрами (data_dir, batch_size, window_length и т.д.)
  - `setup`: разделение на train/val/test
  - `train_dataloader`, `val_dataloader`, `test_dataloader`: создание загрузчиков данных
- Использовать информацию из `segments_info.csv` для создания датасетов

## 2. Стратегия балансировки классов

### 2.1. Взвешивание классов
- Вычислять веса классов на основе соотношения норма/приступы
- Передавать веса в функцию потерь

### 2.2. Undersampling нормальных примеров
- Реализовать случайный undersampling нормальных сегментов
- Целевое соотношение: 25% приступов, 75% нормальной активности

### 2.3. Аугментация данных
- Реализовать аугментации для ЭЭГ данных:
  - Добавление шума
  - Сдвиг по времени
  - Изменение амплитуды

## 3. Создание датасетов с кросс-валидацией по животным

### 3.1. Разделение по животным
- Обеспечить, чтобы одно и то же животное не попадало одновременно в train и val/test
- Реализовать стратегию разделения:
  - Train: 70% животных
  - Val: 15% животных
  - Test: 15% животных

### 3.2. Кросс-валидация
- Реализовать k-fold кросс-валидацию по животным
- Создать отдельные классы для управления кросс-валидацией

## 4. PyTorch Lightning модуль для базовой модели

### 4.1. Базовая модель
- Создать класс `EpilepsyDetector`, наследуемый от `pytorch_lightning.LightningModule`
- Использовать существующую архитектуру `SimpleEEGDetector` или `ImprovedEEGDetector`
- Реализовать методы:
  - `forward`: прямой проход через сеть
  - `training_step`: шаг обучения
  - `validation_step`: шаг валидации
  - `test_step`: шаг тестирования
  - `configure_optimizers`: настройка оптимизатора

### 4.2. Функция потерь
- Использовать взвешенную кросс-энтропию для балансировки классов
- Добавить регуляризацию L2

## 5. Логирование метрик и сохранение результатов

### 5.1. Метрики
- Accuracy, Precision, Recall, F1-Score для класса приступов
- Sensitivity и Specificity
- AUC-ROC
- Матрица ошибок

### 5.2. Логирование
- Использовать встроенный логгер PyTorch Lightning (TensorBoard)
- Сохранять веса модели
- Сохранять конфигурацию эксперимента
- Сохранять предсказания для дальнейшего анализа

## 6. Конфигурационный файл для экспериментов

### 6.1. Структура конфигурации
- Параметры модели (архитектура, гиперпараметры)
- Параметры данных (размер окна, балансировка)
- Параметры обучения (learning rate, batch size, epochs)
- Параметры эксперимента (seed, device, output_dir)

### 6.2. Пример конфигурации
```yaml
# config.yaml
data:
  data_dir: "data/processed"
  window_length: 2000  # 5 секунд при 400 Гц
  batch_size: 64
  train_animal_ratio: 0.7
  val_animal_ratio: 0.15

model:
  input_channels: 4
  num_classes: 2
  dropout_rate: 0.5
  use_improved_architecture: false

training:
  learning_rate: 0.001
  num_epochs: 100
  weight_decay: 1e-4
  class_weights: [1.0, 3.0]  # Веса для нормы и приступов

experiment:
  seed: 42
  device: "cuda"
  output_dir: "experiments/exp_001"
```

## 7. Скрипт для запуска обучения

### 7.1. Основной скрипт
- Загрузка конфигурации
- Инициализация DataModule
- Инициализация модели
- Создание Trainer
- Запуск обучения и валидации

### 7.2. Пример использования
```bash
python train.py --config config.yaml
```

## 8. Структура проекта после реализации

```
epilepsy_detector/
├── data/
│   └── processed/
├── src/
│   ├── data_loading/
│   │   ├── epilepsy_dataset.py
│   │   └── epilepsy_datamodule.py
│   ├── modeling/
│   │   ├── epilepsy_detector.py
│   │   └── lightning_module.py
│   ├── preprocessing/
│   │   └── class_balancer.py
│   └── utils/
├── experiments/
│   └── exp_001/
│       ├── config.yaml
│       ├── checkpoints/
│       └── logs/
├── train.py
└── config.yaml
```

## 9. План реализации

### Этап 1: DataModule (2 дня)
- Реализация EpilepsyDataset
- Реализация EpilepsyDataModule
- Тестирование загрузки данных

### Этап 2: Балансировка классов (1 день)
- Реализация взвешивания классов
- Реализация undersampling
- Реализация аугментаций

### Этап 3: Модель (2 дня)
- Адаптация существующих моделей под Lightning
- Реализация LightningModule
- Тестирование forward pass

### Этап 4: Интеграция и обучение (2 дня)
- Создание скрипта обучения
- Настройка логирования
- Первые эксперименты

### Этап 5: Валидация и тестирование (1 день)
- Реализация кросс-валидации
- Оценка метрик
- Подготовка отчета