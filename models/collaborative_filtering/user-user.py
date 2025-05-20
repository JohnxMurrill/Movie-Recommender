import numpy as np
from scipy import sparse
import os
from joblib import dump, load
from models import logger

class UserUserCollaborativeFiltering:
    """
    User-User Collaborative Filtering using cosine similarity.
    Given a user-item interaction matrix, computes user-user similarities and can recommend items.
    """
    def __init__(self, interaction_matrix):
        """
        interaction_matrix: scipy sparse matrix (users x items)
        """
        self.interaction_matrix = interaction_matrix.tocsr()
        self.user_sim_matrix = None

    def compute_similarity(self):
        """Compute user-user cosine similarity matrix."""
        from sklearn.metrics.pairwise import cosine_similarity
        self.user_sim_matrix = cosine_similarity(self.interaction_matrix)
        return self.user_sim_matrix

    def save_to_cache(self):
        """Save the user similarity matrix to cache."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        dump({
            'user_sim_matrix': self.user_sim_matrix
        }, os.path.join(cache_dir, 'user_user_sim_matrix.joblib'))
        logger.info(f"User-user similarity matrix saved to {os.path.abspath(os.path.join(cache_dir, 'user_user_sim_matrix.joblib'))}")

    def load_from_cache(self):
        """Load the user similarity matrix from cache if available."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        cache_path = os.path.join(cache_dir, 'user_user_sim_matrix.joblib')
        if not os.path.exists(cache_path):
            msg = f"User-user similarity matrix cache not found at {cache_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        data = load(cache_path)
        self.user_sim_matrix = data['user_sim_matrix']
        logger.info(f"User-user similarity matrix loaded from {os.path.abspath(cache_path)}")

    def recommend(self, user_idx, top_n=10, exclude_rated=True):
        """
        Recommend items for a user based on similar users' preferences.
        user_idx: index of the user in the interaction matrix
        top_n: number of recommendations to return
        exclude_rated: if True, do not recommend items the user has already rated
        Returns: list of (item_idx, score) tuples
        """
        if self.user_sim_matrix is None:
            self.compute_similarity()
        # Get similarity scores for the user
        sim_scores = self.user_sim_matrix[user_idx]
        # Weighted sum of other users' ratings
        scores = sim_scores @ self.interaction_matrix
        scores = np.asarray(scores).flatten()
        if exclude_rated:
            user_rated = self.interaction_matrix[user_idx].nonzero()[1]
            scores[user_rated] = -np.inf
        top_items = np.argpartition(scores, -top_n)[-top_n:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]
        return [(item, scores[item]) for item in top_items]
