import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization as NMF
# Save factor matrices after fitting
import os
from joblib import dump, load
from models import logger


# This code implements an Alternating Least Squares (ALS) algorithm for matrix factorization.
class AlternatingLeastSquares:
    def __init__(self, n_factors=50, n_iters=10, reg=0.1, random_state=42):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.random_state = random_state
        self.user_factors = None
        self.item_factors = None

    def fit(self, interaction_matrix):
        np.random.seed(self.random_state)
        n_users, n_items = interaction_matrix.shape
        self.user_factors = np.random.normal(scale=0.1, size=(n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(n_items, self.n_factors))
        reg_eye = self.reg * np.eye(self.n_factors)
        for iteration in range(self.n_iters):
            # Update user factors
            for u in range(n_users):
                item_indices = interaction_matrix[u].indices
                if len(item_indices) == 0:
                    continue
                V = self.item_factors[item_indices]
                ratings = interaction_matrix[u, item_indices].toarray().flatten()
                A = V.T @ V + reg_eye
                b = V.T @ ratings
                self.user_factors[u] = np.linalg.solve(A, b)
            # Update item factors
            for i in range(n_items):
                user_indices = interaction_matrix[:, i].indices
                if len(user_indices) == 0:
                    continue
                U = self.user_factors[user_indices]
                ratings = interaction_matrix[user_indices, i].toarray().flatten()
                A = U.T @ U + reg_eye
                b = U.T @ ratings
                self.item_factors[i] = np.linalg.solve(A, b)
            logger.info(f"Iteration {iteration+1}/{self.n_iters} complete.")

        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        dump({
            'user_factors': self.user_factors,
            'item_factors': self.item_factors
        }, os.path.join(cache_dir, 'als_factors.joblib'))
        logger.info(f"ALS factor matrices saved to {os.path.abspath(os.path.join(cache_dir, 'als_factors.joblib'))}")

    def load_from_cache(self):
        """Load factor matrices from cache if available."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        cache_path = os.path.join(cache_dir, 'als_factors.joblib')
        if not os.path.exists(cache_path):
            msg = f"ALS factor cache not found at {cache_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        factors = load(cache_path)
        self.user_factors = factors['user_factors']
        self.item_factors = factors['item_factors']
        logger.info(f"ALS factor matrices loaded from {os.path.abspath(cache_path)}")

    def predict(self, user_idx, item_idx):
        return self.user_factors[user_idx] @ self.item_factors[item_idx]