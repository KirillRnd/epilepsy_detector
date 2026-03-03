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
  model_name: "UNet1DDetector" 
  class_weights: [1.0, 100.0]  # Веса для нормы и приступов

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

1. **ConvBiGRUDetector** - Conv1d feature extractor + BiGRU temporal smoother
2. **TCNDetector** - Dilated TCN для покадровой детекции приступов
3. **UNet1DDetector** - Encoder-Decoder с skip connections для 1D сегментации

### Сводная таблица результатов

| Модель | val_loss | val_acc | val_f1 | test_acc | test_f1 | test_precision | test_recall |
|--------|----------|---------|--------|----------|---------|----------------|-------------|
| ConvBiGRUDetector | 0.0076 | 0.9704 | 0.1685 | 0.9944 | 0.0053 | 0.0674 | 0.0028 |
| TCNDetector | 0.0086 | 0.9763 | 0.1296 | 0.9944 | 0.0037 | 0.0658 | 0.0019 |
| UNet1DDetector | 0.0086 | 0.9789 | 0.1359 | 0.9945 | 0.0062 | 0.1295 | 0.0032 |
## Метрики оценки качества

Система вычисляет следующие метрики качества:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion matrix