# Стратегия балансировки классов для обучения детектора эпилепсии

## Проблема дисбаланса классов

В задачах детектирования эпилептических приступов обычно наблюдается значительный дисбаланс классов:
- Нормальная активность составляет подавляющее большинство данных
- Эпилептические приступы встречаются редко (обычно менее 1% от общего времени записи)

Этот дисбаланс может привести к следующим проблемам:
1. Модель становится предвзятой в сторону_majority класса
2. Высокая точность может маскировать плохое качество детектирования приступов
3. Модель может просто предсказывать majority класс для всех случаев

## Подходы к балансировке классов

### 1. Undersampling (понижающая дискретизация)

#### Случайный undersampling
```python
import numpy as np
from sklearn.utils import resample

def random_undersampling(X, y, target_ratio=1.0):
    """
    Случайный undersampling для балансировки классов
    
    Параметры:
    X (np.ndarray): признаки
    y (np.ndarray): метки классов
    target_ratio (float): целевое соотношение minority/majority
    
    Возвращает:
    tuple: сбалансированные X и y
    """
    # Разделение на классы
    unique_classes, counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    
    # Индексы для каждого класса
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    # Вычисление целевого количества majority примеров
    target_majority_count = int(len(minority_indices) / target_ratio)
    
    # Случайная выборка из majority класса
    selected_majority_indices = resample(
        majority_indices, 
        replace=False, 
        n_samples=target_majority_count,
        random_state=42
    )
    
    # Объединение индексов
    selected_indices = np.concatenate([minority_indices, selected_majority_indices])
    
    # Создание сбалансированного датасета
    X_balanced = X[selected_indices]
    y_balanced = y[selected_indices]
    
    return X_balanced, y_balanced
```

#### Андерсэмплинг на основе сложности
```python
from sklearn.ensemble import IsolationForest
from sklearn.metrics.pairwise import euclidean_distances

def complexity_based_undersampling(X, y, n_neighbors=5):
    """
    Андерсэмплинг на основе сложности (удаление легких для классификации примеров)
    
    Параметры:
    X (np.ndarray): признаки
    y (np.ndarray): метки классов
    n_neighbors (int): количество соседей для вычисления сложности
    
    Возвращает:
    tuple: сбалансированные X и y
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Разделение на классы
    unique_classes = np.unique(y)
    
    # Для каждого класса отбираем наиболее сложные примеры
    selected_indices = []
    
    for class_label in unique_classes:
        class_indices = np.where(y == class_label)[0]
        class_X = X[class_indices]
        
        if len(class_indices) > 10:  # Только для достаточно больших классов
            # Вычисление сложности как среднего расстояния до соседей другого класса
            other_class_indices = np.where(y != class_label)[0]
            other_class_X = X[other_class_indices]
            
            # KNN для вычисления расстояний
            nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(other_class_X)))
            nbrs.fit(other_class_X)
            distances, _ = nbrs.kneighbors(class_X)
            
            # Сложность как среднее расстояние
            complexity_scores = np.mean(distances, axis=1)
            
            # Отбор наиболее сложных примеров (с наименьшими расстояниями)
            n_keep = max(len(class_indices) // 2, 100)  # Оставляем половину или минимум 100
            keep_indices = np.argsort(complexity_scores)[:n_keep]
            selected_indices.extend(class_indices[keep_indices])
        else:
            selected_indices.extend(class_indices)
    
    return X[selected_indices], y[selected_indices]
```

### 2. Oversampling (повышающая дискретизация)

#### SMOTE (Synthetic Minority Oversampling Technique)
```python
from sklearn.neighbors import NearestNeighbors

def smote_oversampling(X, y, k_neighbors=5, oversampling_ratio=1.0):
    """
    SMOTE oversampling для генерации синтетических примеров minority класса
    
    Параметры:
    X (np.ndarray): признаки
    y (np.ndarray): метки классов
    k_neighbors (int): количество соседей для SMOTE
    oversampling_ratio (float): коэффициент oversampling
    
    Возвращает:
    tuple: сбалансированные X и y
    """
    # Разделение на классы
    unique_classes, counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    minority_X = X[minority_indices]
    
    # Вычисление целевого количества minority примеров
    target_minority_count = int(len(majority_indices) * oversampling_ratio)
    n_synthetic = target_minority_count - len(minority_indices)
    
    if n_synthetic <= 0:
        return X, y
    
    # Генерация синтетических примеров
    synthetic_samples = []
    
    # KNN для поиска соседей
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(minority_X)))
    nbrs.fit(minority_X)
    
    for _ in range(n_synthetic):
        # Случайный выбор примера
        idx = np.random.randint(0, len(minority_X))
        sample = minority_X[idx]
        
        # Поиск соседей
        distances, indices = nbrs.kneighbors([sample])
        neighbor_idx = indices[0][np.random.randint(1, len(indices[0]))]
        neighbor = minority_X[neighbor_idx]
        
        # Генерация синтетического примера
        gap = np.random.random()
        synthetic_sample = sample + gap * (neighbor - sample)
        synthetic_samples.append(synthetic_sample)
    
    # Объединение с исходными данными
    if synthetic_samples:
        X_synthetic = np.array(synthetic_samples)
        y_synthetic = np.full(len(synthetic_samples), minority_class)
        
        X_balanced = np.vstack([X, X_synthetic])
        y_balanced = np.hstack([y, y_synthetic])
    else:
        X_balanced, y_balanced = X, y
    
    return X_balanced, y_balanced
```

#### Аугментация данных
```python
def augment_eeg_data(X, y, augmentation_factor=2, noise_level=0.01):
    """
    Аугментация данных ЭЭГ для балансировки классов
    
    Параметры:
    X (np.ndarray): признаки (окна ЭЭГ)
    y (np.ndarray): метки классов
    augmentation_factor (int): коэффициент аугментации
    noise_level (float): уровень шума для добавления
    
    Возвращает:
    tuple: аугментированные X и y
    """
    # Разделение на классы
    unique_classes, counts = np.unique(y, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]
    
    minority_indices = np.where(y == minority_class)[0]
    minority_X = X[minority_indices]
    
    # Аугментация minority класса
    augmented_samples = []
    augmented_labels = []
    
    for _ in range(augmentation_factor - 1):
        for sample in minority_X:
            # 1. Добавление шума
            noise = np.random.normal(0, noise_level * np.std(sample), sample.shape)
            noisy_sample = sample + noise
            augmented_samples.append(noisy_sample)
            
            # 2. Сдвиг по времени (если это временной ряд)
            if len(sample.shape) > 1:
                shift = np.random.randint(-10, 11)  # Сдвиг от -10 до 10 отсчетов
                shifted_sample = np.roll(sample, shift, axis=1)
                augmented_samples.append(shifted_sample)
            
            # 3. Изменение амплитуды
            amplitude_factor = np.random.uniform(0.8, 1.2)
            amplitude_sample = sample * amplitude_factor
            augmented_samples.append(amplitude_sample)
            
            # Добавляем метки для всех аугментированных примеров
            augmented_labels.extend([minority_class] * 3)
    
    # Объединение с исходными данными
    if augmented_samples:
        X_augmented = np.vstack([X, np.array(augmented_samples)])
        y_augmented = np.hstack([y, np.array(augmented_labels)])
    else:
        X_augmented, y_augmented = X, y
    
    return X_augmented, y_augmented
```

### 3. Взвешивание классов

#### Вычисление весов классов
```python
from sklearn.utils.class_weight import compute_class_weight

def calculate_class_weights(y, method='balanced'):
    """
    Вычисление весов классов для балансировки
    
    Параметры:
    y (np.ndarray): метки классов
    method (str): метод вычисления весов
    
    Возвращает:
    dict: словарь весов классов
    """
    unique_classes = np.unique(y)
    
    if method == 'balanced':
        weights = compute_class_weight('balanced', classes=unique_classes, y=y)
    elif method == 'inverse_frequency':
        counts = np.bincount(y)
        weights = len(y) / (len(unique_classes) * counts)
    else:
        weights = np.ones(len(unique_classes))
    
    return dict(zip(unique_classes, weights))

# Пример использования в модели
class_weights = calculate_class_weights(y_train, method='balanced')
model.fit(X_train, y_train, class_weight=class_weights)
```

## Стратегия балансировки для проекта детектирования эпилепсии

### Рекомендуемый подход:

1. **Комбинированный подход**:
   - Использовать undersampling для majority класса (нормальная активность)
   - Использовать аугментацию для minority класса (приступы)
   - Применить взвешивание классов в функции потерь

2. **Последовательность шагов**:
   ```python
   def balance_dataset_for_epilepsy_detection(X, y, seizure_ratio=0.3):
       """
       Балансировка датасета для детектирования эпилепсии
       
       Параметры:
       X (np.ndarray): признаки
       y (np.ndarray): метки (0 - норма, 1 - приступ)
       seizure_ratio (float): целевое соотношение приступов к норме
       
       Возвращает:
       tuple: сбалансированные X и y
       """
       # 1. Undersampling нормальных примеров
       X_undersampled, y_undersampled = random_undersampling(
           X, y, target_ratio=seizure_ratio
       )
       
       # 2. Аугментация примеров с приступами
       X_balanced, y_balanced = augment_eeg_data(
           X_undersampled, y_undersampled, 
           augmentation_factor=3, noise_level=0.02
       )
       
       return X_balanced, y_balanced
   ```

3. **Параметры балансировки**:
   ```python
   BALANCING_CONFIG = {
       'seizure_ratio': 0.25,  # 25% приступов в сбалансированном датасете
       'augmentation_factor': 3,  # Увеличить minority класс в 3 раза
       'noise_level': 0.01,      # Уровень шума для аугментации
       'undersampling_method': 'random',  # Метод undersampling
       'use_class_weights': True,  # Использовать веса классов
       'class_weight_method': 'balanced'  # Метод вычисления весов
   }
   ```

## Оценка эффективности балансировки

### Метрики для оценки:
```python
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_balancing_effectiveness(y_true, y_pred, y_pred_proba=None):
    """
    Оценка эффективности балансировки классов
    
    Параметры:
    y_true (np.ndarray): истинные метки
    y_pred (np.ndarray): предсказанные метки
    y_pred_proba (np.ndarray): вероятности предсказаний
    
    Возвращает:
    dict: словарь с метриками
    """
    # Базовые метрики
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Матрица ошибок
    cm = confusion_matrix(y_true, y_pred)
    
    # Sensitivity и Specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    
    # Precision и Recall для minority класса
    seizure_precision = report['1']['precision']
    seizure_recall = report['1']['recall']
    seizure_f1 = report['1']['f1-score']
    
    # Balanced Accuracy
    balanced_accuracy = (sensitivity + specificity) / 2
    
    metrics = {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'seizure_precision': seizure_precision,
        'seizure_recall': seizure_recall,
        'seizure_f1': seizure_f1,
        'balanced_accuracy': balanced_accuracy,
        'confusion_matrix': cm.tolist()
    }
    
    # Добавляем ROC AUC если есть вероятности
    if y_pred_proba is not None:
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
            metrics['auc'] = auc
        except:
            metrics['auc'] = None
    
    return metrics
```

## Рекомендации по реализации

1. **Экспериментальный подход**: Протестировать несколько стратегий балансировки
2. **Валидация**: Использовать кросс-валидацию с учетом разбиения по животным
3. **Мониторинг**: Отслеживать метрики для обоих классов отдельно
4. **Адаптация**: Настроить параметры балансировки под конкретный датасет
5. **Документирование**: Сохранять информацию о примененной стратегии балансировки

## Возможные улучшения

1. **Адаптивная балансировка**: Автоматическая настройка параметров балансировки
2. **Фокус на сложных примерах**: Использование методов focal loss
3. **Интеграция с обучением**: Онлайн балансировка в процессе обучения
4. **Многоклассовая балансировка**: Расширение для нескольких типов приступов
5. **Учет временной структуры**: Балансировка с учетом последовательности во времени