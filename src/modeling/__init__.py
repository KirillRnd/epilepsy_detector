"""
Модуль моделирования для системы детектирования эпилепсии
"""

# Импортируем модели для регистрации в model_registry
from .simple_cnn_detector import MinimalEEGDetector_v2, MinimalEEGDetector_ESN

# Импортируем другие важные классы для удобства использования
from .model_registry import get_model_class, list_available_models
from .lightning_epilepsy_detector import EpilepsyDetector_v2