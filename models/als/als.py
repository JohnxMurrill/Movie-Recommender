import numpy as np
import pandas as pd
from sklearn.decomposition import non_negative_factorization as NMF
from implicit.als import AlternatingLeastSquares as ImplicitALS
# Save factor matrices after fitting
import os
from joblib import dump, load


# This code implements an Alternating Least Squares (ALS) algorithm for matrix factorization.
class AlternatingLeastSquares:
    def __init__(self, n_factors=100, n_iters=20, reg=0.03, num_threads=8, loss = True):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.num_threads = num_threads
        self.user_factors = None
        self.item_factors = None
        self.model = None
        self.loss = loss

    def fit(self, interaction_matrix):
        print("Fitting ALS model on interaction matrix with shape:", interaction_matrix.shape)
        model = ImplicitALS(factors=self.n_factors,
                            regularization=self.reg,
                            iterations=self.n_iters,
                            num_threads=self.num_threads,
                            calculate_training_loss= self.loss)
        model.fit(interaction_matrix)
        self.model = model
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        # Save user and item factors
        np.savez(os.path.join(cache_dir, 'als_factors.npz'),
                 user_factors=model.user_factors,
                 item_factors=model.item_factors)
        print(f"ALS factor matrices saved to {os.path.abspath(os.path.join(cache_dir, 'als_factors.npz'))}")

    def load_from_cache(self):
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        cache_path = os.path.join(cache_dir, 'als_factors.npz')
        if not os.path.exists(cache_path):
            msg = f"ALS factor cache not found at {cache_path}"
            print(msg)
            raise FileNotFoundError(msg)
        factors = np.load(cache_path)
        model = ImplicitALS()
        model.user_factors = factors['user_factors']
        model.item_factors = factors['item_factors']
        self.model = model
        print(f"ALS factor matrices loaded from {os.path.abspath(cache_path)}")

    def predict(self, user_idx, item_idx):
        return self.model.user_factors[user_idx] @ self.model.item_factors[item_idx]