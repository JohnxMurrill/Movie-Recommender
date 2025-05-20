import numpy as np
import pandas as pd
import os
from joblib import dump, load
import xgboost as xgb
from models import logger

class XGBoostRecommender:
    """
    XGBoost-based recommender using user, movie, and interaction features.
    Trains and caches a model for fast prediction.
    """
    def __init__(self, X=None, y=None):
        self.model = None
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_path = os.path.join(self.cache_dir, 'xgboost_model.joblib')
        self.X = X
        self.y = y

    def fit(self):
        self.model = xgb.XGBRegressor(n_estimators=100, max_depth=8, learning_rate=0.1, n_jobs=-1)
        self.model.fit(self.X, self.y)
        dump(self.model, self.cache_path)
        logger.info(f"XGBoost model saved to {os.path.abspath(self.cache_path)}")

    def load_from_cache(self):
        if not os.path.exists(self.cache_path):
            raise FileNotFoundError(f"XGBoost model cache not found at {self.cache_path}")
        self.model = load(self.cache_path)
        logger.info(f"XGBoost model loaded from {os.path.abspath(self.cache_path)}")

    def predict(self, user_features, movie_features):
        """
        Predict rating for a user-movie pair or batch.
        user_features: DataFrame or 2D array (n_samples, n_user_features)
        movie_features: DataFrame or 2D array (n_samples, n_movie_features)
        Returns: predictions (array)
        """
        if self.model is None:
            msg = "Model not loaded. Call fit() or load_from_cache() first."
            logger.error(msg)
            raise ValueError(msg)
        X = pd.concat([pd.DataFrame(user_features), pd.DataFrame(movie_features)], axis=1)
        return self.model.predict(X)
