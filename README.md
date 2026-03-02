# Система детектирования эпилепсии у крыс

## Описание проекта

Проект по созданию детектора эпилепсии у крыс включает в себя полный цикл разработки системы машинного обучения для анализа ЭЭГ/ЭКоГ записей у крыс. Система предназначена для автоматического обнаружения эпилептических приступов в записях ЭЭГ/ЭКоГ, сохраненных в форматах .edf.

## Структура проекта

```
epilepsy_detector/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_loading/
│   ├── preprocessing/
│   ├── modeling/
│   └── utils/
├── experiments/
│   └── config.yaml
├── train.py
└── README.md
```

## Установка зависимостей

Для работы проекта необходимо установить следующие зависимости:

```bash
pip install -r requirements.txt
```

Основные зависимости:
- PyTorch
- PyTorch Lightning
- MNE
- NumPy
- Pandas
- Scikit-learn
- TensorBoard

## Подготовка данных

Данные должны быть предварительно обработаны и сохранены в формате .npy. Для этого используйте скрипты из директории `src/utils/`.

Структура обработанных данных:
```
data/processed/
├── <animal_id>/
│   ├── <session_id>/
│   │   ├── processed_signals.npy
│   │   ├── seizure_mask.npy
│   │   ├── conversion_metadata.json
│   │   └── segments_info.csv
└── conversion_summary.csv
```

## Обучение модели

### Конфигурация эксперимента

Перед запуском обучения необходимо настроить параметры в файле `experiments/config.yaml`:

```yaml
# Конфигурационный файл для эксперимента с минимальной архитектурой

# Параметры данных
data:
  data_dir: "data/processed"
  window_length: 2000  # 5 секунд при 400 Гц
  batch_size: 64
  overlap: 0.5
  train_animal_ratio: 0.7
  val_animal_ratio: 0.15
  # Жёсткое разбиение (если задано, то используется вместо случайного)
  train_animals: ["Ati5x1", "Dex1x2NE", "Ati5y2"]  # Список ID животных для обучения
  val_animals: ["Ati5y1"]    # Список ID животных для валидации
  test_animals: ["Dex4x5"]   # Список ID животных для тестирования

# Параметры модели
model:
  input_channels: 3
  window_length: 2000
  num_classes: 2
  dropout_rate: 0.5
  use_minimal_architecture: true
  use_improved_architecture: false
  class_weights: [1.0, 3.0]  # Веса для нормы и приступов

# Параметры обучения
training:
  learning_rate: 0.001
  num_epochs: 50
  weight_decay: 1e-4
  patience: 10

# Параметры эксперимента
experiment:
  seed: 42
  device: "cuda"  # или "cuda"
  output_dir: "experiments/exp_001"
  checkpoint_dir: "experiments/exp_001/checkpoints"
  log_dir: "experiments/exp_001/logs"
```

### Запуск обучения

Для запуска обучения выполните команду:

```bash
python train.py --config experiments/config.yaml
```


## Мониторинг обучения

Для мониторинга процесса обучения можно использовать TensorBoard:

```bash
tensorboard --logdir experiments/exp_001/logs
```

## Архитектура модели

Проект поддерживает две архитектуры моделей:

1. **MinimalEEGDetector_v2** - базовая 1D-CNN архитектура
2. **MinimalEEGDetector_ESN** - ESN-CNN архитектура

## Метрики оценки качества

Система вычисляет следующие метрики качества:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion matrix