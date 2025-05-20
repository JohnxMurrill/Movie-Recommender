import os
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from models.als.als import ALS
from models.collaborative_filtering.user_user import UserUserCollaborativeFiltering
from models.content_filtering.item_item import ItemItemContentFiltering
from models.xgboost.xgboost import XGBoostRecommender

class ModelTrainer:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.data_paths = self.config['data']
        self.trainer_cfg = self.config['trainer']
        self.output_cfg = self.config['output']
        self.models = {}

    def load_data(self, data_key):
        path = self.data_paths[data_key]
        if path.endswith('.csv'):
            return pd.read_csv(path)
        elif path.endswith('.npz'):
            return np.load(path)
        else:
            raise ValueError(f"Unsupported file type for {path}")

    def train_als(self):
        data_key = self.trainer_cfg['als_data']
        data = self.load_data(data_key)
        als = ALS()
        if isinstance(data, np.lib.npyio.NpzFile):
            als.fit(data['arr_0'])
        else:
            raise ValueError("ALS expects a user-movie interaction matrix (npz)")
        als.save_to_cache(os.path.join(self.output_cfg['model_cache_dir'], 'als_factors.joblib'))
        self.models['als'] = als

    def train_user_user(self):
        data_key = self.trainer_cfg['user_user_data']
        data = self.load_data(data_key)
        user_user = UserUserCollaborativeFiltering(data['arr_0'])
        user_user.compute_similarity()
        user_user.save_to_cache(os.path.join(self.output_cfg['model_cache_dir'], 'user_user_sim_matrix.joblib'))
        self.models['user_user'] = user_user

    def train_item_item(self):
        data_key = self.trainer_cfg['item_item_data']
        data = self.load_data(data_key)
        item_item = ItemItemContentFiltering(data)
        item_item.compute_similarity()
        item_item.save_to_cache(os.path.join(self.output_cfg['model_cache_dir'], 'item_item_sim_matrix.joblib'))
        self.models['item_item'] = item_item

    def train_xgboost(self):
        data_key = self.trainer_cfg['xgboost_data']
        data = self.load_data(data_key)
        xgb = XGBoostRecommender()
        if isinstance(data, pd.DataFrame):
            # Assume last column is target if present
            X = data.iloc[:, :-1]
            y = data.iloc[:, -1] if data.shape[1] > 1 else None
            if y is not None:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                xgb.fit(X_train, y_train)
            else:
                raise ValueError("XGBoost expects features and target in the DataFrame")
        else:
            raise ValueError("XGBoost expects a DataFrame (csv)")
        xgb.save_to_cache(os.path.join(self.output_cfg['model_cache_dir'], 'xgboost_model.joblib'))
        self.models['xgboost'] = xgb

    def train_all(self):
        self.train_als()
        self.train_user_user()
        self.train_item_item()
        self.train_xgboost()

# Example usage:
# trainer = ModelTrainer('config.yaml')
# trainer.train_all()
