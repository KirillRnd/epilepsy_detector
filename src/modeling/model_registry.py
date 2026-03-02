"""
Реестр моделей для системы детектирования эпилепсии
"""

from typing import Dict, Type, Callable
import torch.nn as nn

# Глобальный реестр моделей
_MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {}

def register_model(name: str) -> Callable:
    """
    Декоратор для регистрации модели в реестре
    
    Args:
        name: Имя модели для регистрации
    """
    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        if name in _MODEL_REGISTRY:
            raise ValueError(f"Модель с именем '{name}' уже зарегистрирована")
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def get_model_class(name: str) -> Type[nn.Module]:
    """
    Получить класс модели по имени
    
    Args:
        name: Имя модели
        
    Returns:
        Класс модели
        
    Raises:
        ValueError: Если модель с таким именем не найдена
    """
    if name not in _MODEL_REGISTRY:
        available = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(f"Неизвестное имя модели: {name}. Доступные модели: {available}")
    return _MODEL_REGISTRY[name]

def list_available_models() -> list:
    """
    Получить список доступных моделей
    
    Returns:
        Список имен доступных моделей
    """
    return list(_MODEL_REGISTRY.keys())