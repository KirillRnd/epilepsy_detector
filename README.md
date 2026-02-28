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
│   │   └── segments_info.csv
└── conversion_summary.csv
```

## Обучение модели

### Конфигурация эксперимента

Перед запуском обучения необходимо настроить параметры в файле `experiments/config.yaml`:

```yaml
data:
  data_dir: "data/processed"
  window_length: 2000  # 5 секунд при 400 Гц
  batch_size: 64
  overlap: 0.5
  train_animal_ratio: 0.7
  val_animal_ratio: 0.15

model:
  input_channels: 4
  window_length: 2000
  num_classes: 2
  dropout_rate: 0.5
  use_improved_architecture: false
  class_weights: [1.0, 3.0]

training:
  learning_rate: 0.001
  num_epochs: 100
  weight_decay: 1e-4
  patience: 10

experiment:
  seed: 42
  device: "cuda"
  output_dir: "experiments/exp_001"
  checkpoint_dir: "experiments/exp_001/checkpoints"
  log_dir: "experiments/exp_001/logs"
```

### Запуск обучения

Для запуска обучения выполните команду:

```bash
python train.py --config experiments/config.yaml
```

Для запуска обучения с последующим тестированием:

```bash
python train.py --config experiments/config.yaml --test
```

## Мониторинг обучения

Для мониторинга процесса обучения можно использовать TensorBoard:

```bash
tensorboard --logdir experiments/exp_001/logs
```

## Архитектура модели

Проект поддерживает две архитектуры моделей:

1. **SimpleEEGDetector** - базовая 1D-CNN архитектура
2. **ImprovedEEGDetector** - улучшенная архитектура с остаточными блоками

## Метрики оценки качества

Система вычисляет следующие метрики качества:
- Accuracy
- Precision
- Recall
- F1-Score
- Sensitivity (Recall для класса приступов)
- Specificity
- AUC-ROC

## Ожидаемые результаты

Целевые метрики качества:
- Accuracy: > 90%
- F1-Score для класса приступов: > 0.8
- Sensitivity: > 0.85
- Specificity: > 0.95
- AUC-ROC: > 0.95

## Кросс-валидация

Система реализует кросс-валидацию по животным, чтобы обеспечить корректную оценку обобщающей способности модели.

Для запуска кросс-валидации используйте скрипт:

```bash
python run_cv.py --config experiments/config.yaml --n_splits 5
```

## Возможные улучшения

### Краткосрочные:
1. Добавление поддержки других форматов (BIDS, EEGLAB)
2. Расширение набора аугментаций для ЭЭГ данных
3. Реализация ансамблей моделей
4. Добавление механизма внимания

### Долгосрочные:
1. Self-supervised pretraining на больших объемах нормальных данных
2. Комбинирование CNN с рекуррентными слоями (LSTM/GRU)
3. Использование трансформеров для анализа временных последовательностей
4. Разработка системы онлайн-мониторинга приступов