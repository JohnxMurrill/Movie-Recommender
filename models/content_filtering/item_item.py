import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
from joblib import dump, load
from models import logger

class ItemItemContentFiltering:
    """
    Content-based Item-Item Filtering using item feature vectors (e.g., from pre_processed_movies.csv).
    Computes item-item similarities and can recommend similar items for a given item or user profile.

    For users with no existing ratings, they will be matched against the generic user archetypes and get
    recommendations based on the closest archetype.
    """
    def __init__(self, item_features):
        """
        item_features: numpy array or pandas DataFrame of shape (n_items, n_features)
        """
        if hasattr(item_features, 'values'):
            self.item_features = item_features.values
        else:
            self.item_features = item_features
        self.sim_matrix = None

    def compute_similarity(self):
        """Compute item-item cosine similarity matrix."""
        self.sim_matrix = cosine_similarity(self.item_features)
        return self.sim_matrix

    def recommend_similar_items(self, item_idx, top_n=10, exclude_self=True):
        """
        Recommend items most similar to a given item.
        item_idx: index of the item
        top_n: number of recommendations
        exclude_self: if True, do not recommend the item itself
        Returns: list of (item_idx, similarity) tuples
        """
        if self.sim_matrix is None:
            self.compute_similarity()
        sims = self.sim_matrix[item_idx]
        if exclude_self:
            sims[item_idx] = -np.inf
        top_items = np.argpartition(sims, -top_n)[-top_n:]
        top_items = top_items[np.argsort(sims[top_items])[::-1]]
        return [(i, sims[i]) for i in top_items]

    def recommend_for_user_profile(self, user_profile, top_n=10):
        """
        Recommend items for a user profile vector (e.g., average of items they've liked).
        user_profile: 1D array of item feature weights
        top_n: number of recommendations
        Returns: list of (item_idx, score) tuples
        """
        scores = self.item_features @ user_profile
        top_items = np.argpartition(scores, -top_n)[-top_n:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]
        return [(i, scores[i]) for i in top_items]

    def save_to_cache(self):
        """Save the item-item similarity matrix to cache."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        dump({
            'sim_matrix': self.sim_matrix
        }, os.path.join(cache_dir, 'item_item_sim_matrix.joblib'))
        logger.info(f"Item-item similarity matrix saved to {os.path.abspath(os.path.join(cache_dir, 'item_item_sim_matrix.joblib'))}")

    def load_from_cache(self):
        """Load the item-item similarity matrix from cache if available."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        cache_path = os.path.join(cache_dir, 'item_item_sim_matrix.joblib')
        if not os.path.exists(cache_path):
            msg = f"Item-item similarity matrix cache not found at {cache_path}"
            logger.error(msg)
            raise FileNotFoundError(msg)
        data = load(cache_path)
        self.sim_matrix = data['sim_matrix']
        logger.info(f"Item-item similarity matrix loaded from {os.path.abspath(cache_path)}")
