from implicit.cpu.bpr import BayesianPersonalizedRanking
import os
from joblib import dump, load

class BPR:
    """
    Bayesian Personalized Ranking (BPR) model for implicit feedback data.
    This class implements the BPR algorithm using the Implicit library.
    """

    def __init__(self, interaction_matrix = None ,n_factors=50, learning_rate = 0.1, n_iters=60, reg=0.1, num_threads=8):
        self.n_factors = n_factors
        self.n_iters = n_iters
        self.reg = reg
        self.num_threads = num_threads
        self.learning_rate = learning_rate
        self.model = None
        self.interaction_matrix = interaction_matrix
        if self.interaction_matrix is not None:
            self.interaction_matrix.tocsr()  # Ensure the matrix is in CSR format

    def fit(self):
        print("Fitting BPR model on interaction matrix with shape:", self.interaction_matrix.shape)
        model = BayesianPersonalizedRanking(factors=self.n_factors,
                                            regularization=self.reg,
                                            iterations=self.n_iters,
                                            num_threads=self.num_threads,
                                            learning_rate=self.learning_rate)
        model.fit(self.interaction_matrix)
        self.model = model

    def save_to_cache(self):
        """Save the BPR model to cache."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        os.makedirs(cache_dir, exist_ok=True)
        dump(self.model, os.path.join(cache_dir, 'bpr_model.joblib'))
        print(f"BPR model saved to {os.path.abspath(os.path.join(cache_dir, 'bpr_model.joblib'))}")
    
    def load_from_cache(self):
        """Load the BPR model from cache if available."""
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '.cache')
        cache_path = os.path.join(cache_dir, 'bpr_model.joblib')
        if not os.path.exists(cache_path):
            msg = f"BPR model cache not found at {cache_path}"
            print(msg)
            raise FileNotFoundError(msg)
        self.model = load(cache_path)
        print(f"BPR model loaded from {os.path.abspath(cache_path)}")

    def recommend(self, user_idx, top_n=10):
        """
        Recommend items for a user based on the BPR model.
        user_idx: index of the user in the interaction matrix
        top_n: number of recommendations to return
        Returns: list of (item_idx, score) tuples
        """
        scores = self.model.recommend(user_idx, self.interaction_matrix)
        return [(int(item), float(score)) for item, score in scores[:top_n]]