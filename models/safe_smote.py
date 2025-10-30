# filepath: src/models/safe_smote.py
from imblearn.over_sampling import SMOTENC
from sklearn.base import BaseEstimator
import numpy as np

class SafeSMOTE(BaseEstimator):
    """SMOTE wrapper that adapts k_neighbors to avoid the training size issue."""
    
    def __init__(self, categorical_features, random_state=42, k_neighbors=5):
        self.categorical_features = categorical_features
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self._smote = None
    
    def fit_resample(self, X, y):
        # Calculate safe k_neighbors based on actual data size
        min_class_count = np.bincount(y).min()
        safe_k_neighbors = min(self.k_neighbors, max(1, min_class_count - 1))
        
        print(f"Using k_neighbors={safe_k_neighbors} for fold with min_class={min_class_count}")
        
        self._smote = SMOTENC(
            categorical_features=self.categorical_features,
            random_state=self.random_state,
            k_neighbors=safe_k_neighbors
        )
        
        return self._smote.fit_resample(X, y)