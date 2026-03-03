"""
Модуль моделирования для системы детектирования эпилепсии
"""

# Импортируем модели для регистрации в model_registry
from .simple_cnn_detector import MinimalEEGDetector_v2, MinimalEEGDetector_ESN
from .ConvBiGRUDetector import ConvBiGRUDetector
from .ConvBiGRUDetector_v2 import ConvBiGRUDetector_v2
from .ConvBiGRUDetector_v3 import ConvBiGRUDetector_v3
from .MSConvBiGRUDetector import MSConvBiGRUDetector
from .RDSCBiGRUDetector import RDSCBiGRUDetector
from .TCNDetector import TCNDetector
from .UNet1DDetector import UNet1DDetector

# Импортируем другие важные классы для удобства использования
from .model_registry import get_model_class, list_available_models
from .lightning_epilepsy_detector import EpilepsyDetector_v2