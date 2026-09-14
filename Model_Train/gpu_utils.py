"""
GPU Detection Utility for XGBoost Training
Automatically detects CUDA availability and returns appropriate XGBoost parameters.
"""

import warnings
warnings.filterwarnings('ignore')


def is_xgboost_gpu_available() -> bool:
    """
    Check if XGBoost can actually use GPU training.
    This tests by running a tiny training job, which is the only reliable way.
    
    Returns:
        bool: True if XGBoost GPU training works, False otherwise
    """
    try:
        import xgboost as xgb
        import numpy as np
        
        # Create tiny dummy data
        X = np.random.rand(10, 2)
        y = np.random.rand(10)
        
        # Try to train with gpu_hist
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'tree_method': 'gpu_hist',
            'device': 'cuda',
            'objective': 'reg:squarederror',
            'verbosity': 0
        }
        
        # If this succeeds, GPU is available
        xgb.train(params, dtrain, num_boost_round=1)
        return True
        
    except Exception:
        # gpu_hist not available (XGBoost built without CUDA)
        return False


def get_xgb_device_params() -> dict:
    """
    Get XGBoost device parameters based on actual GPU availability.
    Tests XGBoost's ability to use GPU, not just if a GPU exists.
    
    Returns:
        dict: XGBoost parameters for GPU or CPU training
    """
    if is_xgboost_gpu_available():
        print("🚀 GPU training enabled! Using CUDA for XGBoost.")
        return {
            'tree_method': 'gpu_hist',
            'device': 'cuda'
        }
    else:
        print("💻 Using CPU for XGBoost training (GPU not available or XGBoost built without CUDA).")
        return {
            'tree_method': 'auto',  # Let XGBoost decide (safest default)
            'device': 'cpu'
        }


# Cache the result to avoid repeated detection
_DEVICE_PARAMS = None

def get_cached_device_params() -> dict:
    """
    Get cached XGBoost device parameters (detects once, reuses).
    
    Returns:
        dict: XGBoost parameters for GPU or CPU training
    """
    global _DEVICE_PARAMS
    if _DEVICE_PARAMS is None:
        _DEVICE_PARAMS = get_xgb_device_params()
    return _DEVICE_PARAMS


if __name__ == "__main__":
    # Test the detection
    print("Testing XGBoost GPU detection...")
    params = get_xgb_device_params()
    print(f"Device params: {params}")
