# Структура проекта для детектора эпилепсии у крыс

## Основная структура каталогов

```
epilepsy_detector/
├── data/
│   ├── raw/
│   │   ├── <animal_id>/
│   │   │   ├── <session_id>/
│   │   │   │   ├── recording.edf
│   │   │   │   ├── recording.adicht
│   │   │   │   └── annotations.txt
│   │   │   └── metadata.csv
│   ├── processed/
│   │   ├── <animal_id>/
│   │   │   ├── <session_id>/
│   │   │   │   ├── processed_signals.npy
│   │   │   │   └── windows_labels.csv
│   │   └── dataset_metadata.csv
│   └── labels/
│       ├── <animal_id>/
│       │   ├── <session_id>.txt
│       └── all_seizures.csv
├── src/
│   ├── data_loading/
│   ├── preprocessing/
│   ├── modeling/
│   ├── evaluation/
│   └── utils/
├── notebooks/
├── experiments/
│   ├── experiment_001_baseline/
│   └── experiment_002_improved/
├── results/
├── docs/
└── requirements.txt
```

## Подробное описание структуры

### data/raw/
Содержит исходные необработанные данные:
- Каждое животное в отдельной папке с идентификатором (например, `Dex1y2`)
- Каждая сессия записи в отдельной подпапке (например, `BL_10May`)
- Файлы записи (.edf или .adicht) и файлы разметки (.txt)

### data/processed/
Содержит предобработанные данные:
- Предобработанные сигналы в формате .npy для быстрой загрузки
- Метки для временных окон
- Метаданные датасета

### data/labels/
Содержит файлы разметки:
- Текстовые файлы с разметкой приступов для каждой сессии
- Сводная таблица всех приступов

### src/
Исходный код проекта:
- `data_loading/` - модули для загрузки данных
- `preprocessing/` - модули для предобработки сигналов
- `modeling/` - реализация моделей
- `evaluation/` - модули для оценки качества
- `utils/` - вспомогательные функции

### notebooks/
Jupyter ноутбуки для исследовательского анализа и визуализации

### experiments/
Конфигурации и результаты экспериментов:
- Каждый эксперимент в отдельной папке
- Конфигурационные файлы
- Логи обучения
- Сохраненные модели

### results/
Результаты экспериментов и метрики

### docs/
Документация проекта

### requirements.txt
Зависимости проекта