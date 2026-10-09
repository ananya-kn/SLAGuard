import xgboost as xgb
import numpy as np
from typing import Dict, Any
from .base import BaseModel
from .feature_preprocessor import FeaturePreprocessor


class XGBoostModel(BaseModel):
    """XGBoost model adapter for regression and classification"""
    
    def __init__(self, model_name: str = ""):
        super().__init__()
        self.model_name = model_name  # e.g., "duration_predictor"
    
    def load(self, path: str) -> None:
        self.model = xgb.Booster()
        self.model.load_model(path)
        self.is_loaded = True
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        # Get feature names based on model type
        feature_names = FeaturePreprocessor.get_feature_names(self.model_name)
        dmatrix = xgb.DMatrix(features, feature_names=feature_names)
        return self.model.predict(dmatrix)
    
    def preprocess(self, raw_input: Dict[str, Any]) -> np.ndarray:
        """Route to appropriate preprocessor based on model type"""
        if 'duration' in self.model_name.lower():
            return FeaturePreprocessor.preprocess_duration(raw_input)
        elif 'routing' in self.model_name.lower():
            return FeaturePreprocessor.preprocess_routing(raw_input)
        elif 'sla' in self.model_name.lower() or 'breach' in self.model_name.lower():
            return FeaturePreprocessor.preprocess_sla_breach(raw_input)
        else:
            # Fallback to duration preprocessing
            return FeaturePreprocessor.preprocess_duration(raw_input)
    
    @property
    def model_type(self) -> str:
        return "xgboost"
