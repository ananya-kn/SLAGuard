import os
import yaml
from typing import Dict, Optional, Any
from models.base import BaseModel
from models.xgboost_model import XGBoostModel

class ModelRegistry:
    """Central registry for managing and loading ML models"""
    
    MODEL_TYPES = {
        "xgboost": XGBoostModel
    }
    
    def __init__(self, registry_path: str):
        self.registry_path = registry_path
        self.weights_dir = os.path.dirname(registry_path)
        self.config: Dict = {}
        self.loaded_models: Dict[str, BaseModel] = {}
        self._load_registry()
    
    def _load_registry(self):
        with open(self.registry_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def list_models(self) -> Dict[str, Any]:
        """List all available models and their versions"""
        return self.config.get("models", {})
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> BaseModel:
        """Load and return a model by name and version"""
        models = self.config.get("models", {})
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        model_config = models[model_name]
        version = version or model_config.get("default_version")
        
        if version not in model_config.get("versions", {}):
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        cache_key = f"{model_name}:{version}"
        if cache_key in self.loaded_models:
            return self.loaded_models[cache_key]
        
        version_config = model_config["versions"][version]
        model_type = version_config["type"]
        model_path = os.path.join(self.weights_dir, version_config["path"])
        
        if model_type not in self.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model = self.MODEL_TYPES[model_type](model_name=model_name)
        
        # Only load if file exists (graceful handling for missing weights)
        if os.path.exists(model_path):
            model.load(model_path)
        else:
            print(f"Warning: Model weights not found at {model_path}")
        
        self.loaded_models[cache_key] = model
        return model
    
    def predict(self, model_name: str, features: Dict[str, Any], version: Optional[str] = None) -> Any:
        """Run prediction using specified model"""
        model = self.get_model(model_name, version)
        preprocessed = model.preprocess(features)
        return model.predict(preprocessed)
