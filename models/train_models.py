import os
import yaml
import tqdm
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
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
        batch_size = 1024
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = pd.read_csv("../data/ml-32m/ratings.csv").sample(frac=0.1, random_state=42)

        # Map userId and movieId to contiguous indices
        user_ids = data['userId'].unique()
        movie_ids = data['movieId'].unique()
        user2idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        movie2idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        num_users = len(user_ids)
        num_items = len(movie_ids)
        embedding_dim = 32  # or set from config

        # Prepare dataset as tensors
        data['user_idx'] = data['userId'].map(user2idx)
        data['movie_idx'] = data['movieId'].map(movie2idx)
        user_tensor = torch.tensor(data['user_idx'].values, dtype=torch.long)
        item_tensor = torch.tensor(data['movie_idx'].values, dtype=torch.long)
        rating_tensor = torch.tensor(data['rating'].values, dtype=torch.float)

        # Create TensorDataset and DataLoader
        dataset = torch.utils.data.TensorDataset(user_tensor, item_tensor, rating_tensor)
        train_length = int(len(dataset) * 0.8)
        valid_length = int(len(dataset) * 0.1)
        test_length = len(dataset) - train_length - valid_length
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            dataset, (train_length, valid_length, test_length))
        train_data_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True
        )

        # Initialize model, optimizer, and loss
        model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim).to(device)
        optimizer = Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()  # For explicit ratings; use nn.BCELoss() for implicit

        # Train the model
        trainer = NCFTrainer(model, optimizer, criterion, device)
        trainer.train(train_data_loader, num_epochs=20)
        model.save_to_cache()

    
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
        # self.train_als()
        # self.train_bpr()
        # self.train_lmf()
        self.fit_NCF()

if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer('config.yaml')
    trainer.train_all()
