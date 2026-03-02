import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
from typing import Dict, List, Tuple, Optional


class SimpleEEGDetector(nn.Module):
    """
    Простой 1D-CNN детектор эпилептических приступов
    """
    
    def __init__(self, 
                 input_channels: int = 4,
                 window_length: int = 5120,  # 5 секунд при 1024 Гц
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):
        """
        Инициализация модели
        
        Параметры:
        input_channels (int): количество каналов ЭЭГ
        window_length (int): длина временного окна в отсчетах
        num_classes (int): количество классов (2 для бинарной классификации)
        dropout_rate (float): вероятность dropout
        """
        super(SimpleEEGDetector, self).__init__()
        
        self.input_channels = input_channels
        self.window_length = window_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Сверточные слои
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=64, stride=8, padding=32)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=32, stride=4, padding=16)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=16, stride=2, padding=8)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Вычисление размера после сверток
        conv_output_size = self._calculate_conv_output_size()
        
        # Полносвязные слои
        self.fc1 = nn.Linear(128 * conv_output_size, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(128, num_classes)
        
        # Инициализация весов
        self._initialize_weights()
    
    def _calculate_conv_output_size(self):
        """
        Вычисление размера выхода после сверточных слоев
        """
        # Размер после каждого слоя
        size = self.window_length
        
        # Conv1: kernel=64, stride=8, padding=32
        size = (size + 2 * 32 - 64) // 8 + 1
        
        # Conv2: kernel=32, stride=4, padding=16
        size = (size + 2 * 16 - 32) // 4 + 1
        
        # Conv3: kernel=16, stride=2, padding=8
        size = (size + 2 * 8 - 16) // 2 + 1
        
        return size
    
    def _initialize_weights(self):
        """
        Инициализация весов модели
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Параметры:
        x (torch.Tensor): входные данные (batch_size, channels, time)
        
        Возвращает:
        torch.Tensor: выходные логиты
        """
        # Сверточные слои с активацией и нормализацией
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        
        # Преобразование для полносвязных слоев
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


class ResidualBlock1D(nn.Module):
    """
    Остаточный блок для 1D-CNN
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """
        Инициализация остаточного блока
        
        Параметры:
        in_channels (int): количество входных каналов
        out_channels (int): количество выходных каналов
        kernel_size (int): размер ядра свертки
        """
        super(ResidualBlock1D, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Преобразование размерности при необходимости
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        """
        Прямой проход через остаточный блок
        """
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        
        return out


class ImprovedEEGDetector(nn.Module):
    """
    Улучшенный 1D-CNN детектор с остаточными блоками
    """
    
    def __init__(self, 
                 input_channels: int = 4,
                 window_length: int = 5120,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3):
        """
        Инициализация улучшенной модели
        """
        super(ImprovedEEGDetector, self).__init__()
        
        self.input_channels = input_channels
        self.window_length = window_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Начальные сверточные слои
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=16, stride=2, padding=8)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Остаточные блоки
        self.res_block1 = ResidualBlock1D(32, 64, kernel_size=15)
        self.res_block2 = ResidualBlock1D(64, 128, kernel_size=11)
        self.res_block3 = ResidualBlock1D(128, 256, kernel_size=7)
        
        # Адаптивный пулинг для фиксированного размера
        self.adaptive_pool = nn.AdaptiveAvgPool1d(10)
        
        # Полносвязные слои
        self.fc1 = nn.Linear(256 * 10, 512)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        """
        Прямой проход через улучшенную сеть
        """
        # Начальные свертки
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Остаточные блоки
        x = self.res_block1(x)
        x = F.max_pool1d(x, 2)
        
        x = self.res_block2(x)
        x = F.max_pool1d(x, 2)
        
        x = self.res_block3(x)
        x = F.max_pool1d(x, 2)
        
        # Адаптивный пулинг
        x = self.adaptive_pool(x)
        
        # Преобразование для полносвязных слоев
        x = x.view(x.size(0), -1)
        
        # Полносвязные слои
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        
        x = self.fc3(x)
        
        return x


def weighted_cross_entropy_loss(outputs, targets, class_weights=None):
    """
    Взвешенная кросс-энтропия для балансировки классов
    
    Параметры:
    outputs (torch.Tensor): выходы модели
    targets (torch.Tensor): истинные метки
    class_weights (torch.Tensor): веса классов
    
    Возвращает:
    torch.Tensor: значение функции потерь
    """
    if class_weights is not None:
        return F.cross_entropy(outputs, targets, weight=class_weights)
    else:
        return F.cross_entropy(outputs, targets)


def train_epoch(model, dataloader, optimizer, criterion, device, class_weights=None):
    """
    Обучение модели в течение одной эпохи
    
    Параметры:
    model (nn.Module): модель для обучения
    dataloader (DataLoader): загрузчик обучающих данных
    optimizer (Optimizer): оптимизатор
    criterion (function): функция потерь
    device (torch.device): устройство для вычислений
    class_weights (torch.Tensor): веса классов
    
    Возвращает:
    float: среднее значение функции потерь
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        
        # Обнуление градиентов
        optimizer.zero_grad()
        
        # Прямой проход
        outputs = model(data)
        
        # Вычисление потерь
        loss = criterion(outputs, targets, class_weights)
        
        # Обратный проход
        loss.backward()
        
        # Обновление весов
        optimizer.step()
        
        # Статистика
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, dataloader, criterion, device, class_weights=None):
    """
    Валидация модели
    
    Параметры:
    model (nn.Module): модель для валидации
    dataloader (DataLoader): загрузчик валидационных данных
    criterion (function): функция потерь
    device (torch.device): устройство для вычислений
    class_weights (torch.Tensor): веса классов
    
    Возвращает:
    tuple: (средняя потеря, точность, предсказания, истинные метки)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            
            # Прямой проход
            outputs = model(data)
            
            # Вычисление потерь
            loss = criterion(outputs, targets, class_weights)
            
            # Статистика
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Сохранение предсказаний для метрик
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_predictions, all_targets


class MinimalEEGDetector(nn.Module):
    """
    Минимальная 1D-CNN модель для тестирования с ~10,000 параметрами
    """
    
    def __init__(self,
                 input_channels: int = 4,
                 window_length: int = 2000,  # 5 секунд при 400 Гц
                 num_classes: int = 2):
        """
        input_channels (int): количество каналов ЭЭГ
        window_length (int): длина временного окна в отсчетах
        num_classes (int): количество классов (2 для бинарной классификации)
        """
        super(MinimalEEGDetector, self).__init__()
        
        self.input_channels = input_channels
        self.window_length = window_length
        self.num_classes = num_classes
        
        # Очень простая сверточная архитектура
        self.conv1 = nn.Conv1d(input_channels, 12, kernel_size=16, stride=6, padding=8)
        self.bn1 = nn.BatchNorm1d(12)
        
        self.conv2 = nn.Conv1d(12, 24, kernel_size=8, stride=3, padding=4)
        self.bn2 = nn.BatchNorm1d(24)
        
        # Вычисление размера после сверток и пулингов
        conv_output_size = self._calculate_conv_output_size()
        
        # Минимальный полносвязный слой
        # ИСПРАВЛЕНИЕ: conv2 выдает 24 канала, поэтому умножаем conv_output_size на 24
        self.fc = nn.Linear(24 * conv_output_size, num_classes)
        # Инициализация весов
        self._initialize_weights()
    
    def _calculate_conv_output_size(self):
        """
        Вычисление размера выхода после сверточных слоев и слоев пулинга
        """
        # Размер после каждого слоя
        size = self.window_length
        
        # Conv1: kernel=16, stride=6, padding=8
        size = (size + 2 * 8 - 16) // 6 + 1
        # ИСПРАВЛЕНИЕ: Учитываем первый max_pool1d(x, 2)
        size = size // 2
        
        # Conv2: kernel=8, stride=3, padding=4
        size = (size + 2 * 4 - 8) // 3 + 1
        # ИСПРАВЛЕНИЕ: Учитываем второй max_pool1d(x, 2)
        size = size // 2
        
        return size
    
    def _initialize_weights(self):
        """
        Инициализация весов модели
        """
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Прямой проход через сеть
        
        Параметры:
        x (torch.Tensor): входные данные (batch_size, channels, time)
        
        Возвращает:
        torch.Tensor: выходные логиты
        """
        # Сверточные слои с активацией и нормализацией
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        # Преобразование для полносвязного слоя
        x = x.view(x.size(0), -1)
        
        # Выходной слой
        x = self.fc(x)
        
        return x

class MinimalEEGDetector_v2(nn.Module):
    """
    Минимальная fully-conv 1D-CNN для детекции seizure по каждому семплу окна.
    Вход:  (B, C=3, T=2000)
    Выход: (B, T=2000) logits
    """

    def __init__(self, input_channels: int = 3, hidden: int = 16, dropout: float = 0.1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
        )

        # 1x1 conv: превращаем признаки в 1 логит на каждый момент времени
        self.head = nn.Conv1d(hidden // 2, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        returns logits: (B, T)
        """
        x = self.features(x)          # (B, H, T)
        x = self.head(x)              # (B, 1, T)
        x = x.squeeze(1)              # (B, T)
        return x

class ESNLayer(nn.Module):
    def __init__(
        self,
        input_size: int,        # C
        reservoir_size: int,    # H
        leaking_rate: float = 1.0,
        spectral_radius: float = 0.9,
        input_scale: float = 1.0,
        bias: bool = True,
        train_reservoir: bool = False,
    ):
        super().__init__()

        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.leaking_rate = leaking_rate

        # W_in: (H, C)
        self.W_in = nn.Parameter(
            torch.empty(reservoir_size, input_size),
            requires_grad=train_reservoir
        )
        # W_res: (H, H)
        self.W_res = nn.Parameter(
            torch.empty(reservoir_size, reservoir_size),
            requires_grad=train_reservoir
        )
        if bias:
            self.b = nn.Parameter(torch.zeros(reservoir_size),
                                  requires_grad=train_reservoir)
        else:
            self.register_parameter("b", None)

        self.input_scale = input_scale
        self.spectral_radius = spectral_radius

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        # случайная инициализация
        nn.init.normal_(self.W_in, mean=0.0, std=self.input_scale)
        nn.init.normal_(self.W_res, mean=0.0, std=1.0)

        # нормируем на заданный spectral_radius
        # оцениваем спектральный радиус по максимальному сингулярному значению
        # (приближение, но на практике достаточно)
        u, s, v = torch.svd(self.W_res)
        if s[0] > 0:
            self.W_res.data *= self.spectral_radius / s[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        return: states: (B, H, T)
        """
        B, C, T = x.shape
        device = x.device

        # (B, T, C)
        x_t = x.permute(0, 2, 1)

        # начальное состояние резервуара
        h = torch.zeros(B, self.reservoir_size, device=device)

        states = []

        for t in range(T):
            u_t = x_t[:, t, :]  # (B, C)

            pre = F.linear(u_t, self.W_in) + F.linear(h, self.W_res)
            if self.b is not None:
                pre = pre + self.b

            pre = torch.tanh(pre)

            # leaky integration
            h = (1.0 - self.leaking_rate) * h + self.leaking_rate * pre

            states.append(h.unsqueeze(2))  # (B, H, 1)

        # (B, H, T)
        states = torch.cat(states, dim=2)
        return states

class MinimalEEGDetector_ESN(nn.Module):
    def __init__(self,
                 input_channels: int = 3,
                 esn_hidden: int = 32,
                 conv_hidden: int = 16,
                 dropout: float = 0.1):
        super().__init__()

        self.esn = ESNLayer(
            input_size=input_channels,
            reservoir_size=esn_hidden,
            leaking_rate=0.3,
            spectral_radius=0.9,
            input_scale=0.5,
            bias=True,
            train_reservoir=False,  # сначала можно заморозить
        )

        hidden = conv_hidden
        self.features = nn.Sequential(
            nn.Conv1d(esn_hidden, hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Conv1d(hidden, hidden // 2, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Conv1d(hidden // 2, 1, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.esn(x)          # (B, esn_hidden, T)
        x = self.features(x)     # (B, H, T)
        x = self.head(x)         # (B, 1, T)
        x = x.squeeze(1)         # (B, T)
        return x

def train_model(model, train_loader, val_loader,
                num_epochs=50, learning_rate=0.001,
                class_weights=None, device='cpu'):
    """
    Полный цикл обучения модели
    
    Параметры:
    model (nn.Module): модель для обучения
    train_loader (DataLoader): загрузчик обучающих данных
    val_loader (DataLoader): загрузчик валидационных данных
    num_epochs (int): количество эпох обучения
    learning_rate (float): начальная скорость обучения
    class_weights (torch.Tensor): веса классов
    device (str): устройство для обучения
    
    Возвращает:
    dict: история обучения
    """
    # Перенос модели на устройство
    model = model.to(device)
    
    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Планировщик скорости обучения
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    
    # История обучения
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Начало обучения на устройстве: {device}")
    print(f"Количество эпох: {num_epochs}")
    print(f"Скорость обучения: {learning_rate}")
    
    for epoch in range(num_epochs):
        # Обучение
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, weighted_cross_entropy_loss, 
            device, class_weights
        )
        
        # Валидация
        val_loss, val_acc, _, _ = validate_epoch(
            model, val_loader, weighted_cross_entropy_loss, 
            device, class_weights
        )
        
        # Обновление планировщика
        scheduler.step(val_loss)
        
        # Сохранение истории
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Сохранение лучшей модели
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        # Вывод статистики
        if (epoch + 1) % 10 == 0:
            print(f'Эпоха [{epoch+1}/{num_epochs}]')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # Загрузка лучшего состояния модели
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Загружено лучшее состояние модели с val_loss = {best_val_loss:.4f}")
    
    return history


def evaluate_model(model, test_loader, device='cpu'):
    """
    Оценка качества модели на тестовой выборке
    
    Параметры:
    model (nn.Module): обученная модель
    test_loader (DataLoader): загрузчик тестовых данных
    device (str): устройство для вычислений
    
    Возвращает:
    dict: словарь с метриками
    """
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            
            # Прямой проход
            outputs = model(data)
            probabilities = F.softmax(outputs, dim=1)
            
            # Предсказания
            _, predicted = outputs.max(1)
            
            # Сохранение результатов
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Вычисление метрик
    predictions = np.array(all_predictions)
    targets = np.array(all_targets)
    probabilities = np.array(all_probabilities)
    
    # Матрица ошибок
    cm = confusion_matrix(targets, predictions)
    
    # Метрики классификации
    report = classification_report(targets, predictions, output_dict=True)
    
    # ROC AUC (для бинарной классификации)
    try:
        if probabilities.shape[1] == 2:
            auc = roc_auc_score(targets, probabilities[:, 1])
        else:
            auc = roc_auc_score(targets, probabilities, multi_class='ovr')
    except:
        auc = None
    
    # Sensitivity и Specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    metrics = {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'] if '1' in report else 0,
        'recall': report['1']['recall'] if '1' in report else 0,
        'f1_score': report['1']['f1-score'] if '1' in report else 0,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': auc,
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


# Рекомендуемые параметры модели
MODEL_CONFIG = {
    'input_channels': 4,           # Количество каналов ЭЭГ
    'window_length': 5120,           # Длина окна (5 сек при 1024 Гц)
    'num_classes': 2,               # Бинарная классификация
    'dropout_rate': 0.5,             # Вероятность dropout
    'learning_rate': 0.001,         # Скорость обучения
    'batch_size': 64,                # Размер батча
    'num_epochs': 100,              # Количество эпох
    'weight_decay': 1e-4,           # Регуляризация L2
    'patience': 10                  # Терпение для early stopping
}


# Рекомендуемые параметры улучшенной модели
IMPROVED_MODEL_CONFIG = {
    'input_channels': 4,
    'window_length': 5120,
    'num_classes': 2,
    'dropout_rate': 0.3,
    'learning_rate': 0.0005,
    'batch_size': 32,
    'num_epochs': 150,
    'weight_decay': 1e-5,
    'patience': 15
}