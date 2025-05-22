import os
import yaml
import tqdm
import pandas as pd
import numpy as np
import torch as nn
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import sparse
from als.als import AlternatingLeastSquares
from bayesian_personalized_rankings.bpr import BPR
from logistic_matrix_factorization.lmf import LMF
from neural_collaborative_filtering.ncf import NeuralCollaborativeFiltering, NCFTrainer

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
        csr_matrix = sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
        als = AlternatingLeastSquares()
        if isinstance(data, np.lib.npyio.NpzFile):
            als.fit(csr_matrix)
        else:
            raise ValueError("ALS expects a user-movie interaction matrix (npz)")
        self.models['als'] = als
    
    def fit_NCF(self):
        device_cpu = "cpu"
        learning_rate = 0.001
        weight_decay = 1e-4
        batch_size = 64
        epochs = 20
        model_name = "ncf"
        device = nn.device(device_cpu)
        data = pd.read_csv("../data/ml-32m/ratings.csv")
        train_length = int(len(data) * 0.8)
        valid_length = int(len(data) * 0.1)
        test_length = len(data) - train_length - valid_length
        train_dataset, valid_dataset, test_dataset = nn.utils.data.random_split(
            data, (train_length, valid_length, test_length))
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=4)
        test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)
        

    
    def train_bpr(self):
        data_key = self.trainer_cfg['bpr_data']
        data = self.load_data(data_key)
        if isinstance(data, np.lib.npyio.NpzFile):
            csr_matrix = sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
            bpr = BPR(csr_matrix)
        else:
            raise ValueError("BPR expects a user-movie interaction matrix (npz)")
        bpr.fit()
        bpr.save_to_cache()
        self.models['bpr'] = bpr
    
    def train_lmf(self):
        data_key = self.trainer_cfg['lmf_data']
        data = self.load_data(data_key)
        if isinstance(data, np.lib.npyio.NpzFile):
            csr_matrix = sparse.csr_matrix((data['data'], data['indices'], data['indptr']), shape=data['shape'])
            lmf = LMF(csr_matrix)
        else:
            raise ValueError("LMF expects a user-movie interaction matrix (npz)")
        lmf.fit()
        lmf.save_to_cache()
        self.models['lmf'] = lmf

    def train_all(self):
        self.train_als()
        # self.train_user_user()
        # self.train_item_item()
        self.train_bpr()
        self.train_lmf()

# Example usage:
trainer = ModelTrainer('config.yaml')
trainer.train_all()
