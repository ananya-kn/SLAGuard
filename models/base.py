from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any

class BaseModel(ABC):
    """Abstract base class for all prediction models"""
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights from file"""
        pass
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Make predictions on preprocessed features"""
        pass
    
    @abstractmethod
    def preprocess(self, raw_input: Dict[str, Any]) -> np.ndarray:
        """Preprocess raw input into model-ready features"""
        pass
    
    @property
    @abstractmethod
    def model_type(self) -> str:
        """Return the model type identifier"""
        pass
