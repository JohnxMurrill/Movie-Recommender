import numpy as np
import pandas as pd
from scipy import sparse
import torch
from models.als.als import AlternatingLeastSquares
from models.bayesian_personalized_rankings.bpr import BPR
from models.logistic_matrix_factorization.lmf import LMF
from models.neural_collaborative_filtering.ncf import NeuralCollaborativeFiltering

def get_test_data():
    # Test user-movie interactions
    test_user_movie_interactions = sparse.load_npz("../data/processed_data/user_movie_interactions.npz")
    return test_user_movie_interactions

def test_XGBoost_predictions():
    pass

def test_ALS_predictions():
    print("Testing Alternating Least Squares")
    test_ALS = AlternatingLeastSquares()
    test_ALS.load_from_cache()
    movies = pd.read_csv("../data/ml-32m/movies.csv")
    assert isinstance(test_ALS, AlternatingLeastSquares)
    test_data = get_test_data()
    test_user = 0
    test_item = 2353 # 2353 - 4.5, 2005 - 1.0, 1198 - 3.5
    prediction = test_ALS.predict(test_user, test_item)
    print(test_data.shape)
    print("User, item: ", test_user, test_item)
    print("Prediction: ", prediction)
    print(f"Top 10 recommendations for user {test_user}: ")
    rec_indices, rec_scores = test_ALS.model.recommend(test_user, test_data[test_user], N=10, filter_already_liked_items=False)
    for idx, score in zip(rec_indices, rec_scores):
        movie_id = movies.iloc[idx]['movieId']
        movie_title = movies.iloc[idx+1]['title']
        print(f"MovieId: {movie_id}, Title: {movie_title}, Score: {score:.4f}")
    return

def test_bpr_predictions():
    print("Testing Bayesian Personalized Rankings")
    test_data = get_test_data()
    test_bpr = BPR(interaction_matrix = test_data)
    test_bpr.load_from_cache()
    assert isinstance(test_bpr, BPR)
    movies = pd.read_csv("../data/ml-32m/movies.csv")
    test_user = 0
    print(f"Top 10 recommendations for user {test_user}: ")
    rec_indices, rec_scores = test_bpr.model.recommend(test_user, test_data, N=10, filter_already_liked_items=False)
    for idx, score in zip(rec_indices, rec_scores):
        movie_id = movies.iloc[idx+1]['movieId']
        movie_title = movies.iloc[idx]['title']
        print(f"MovieId: {movie_id}, Title: {movie_title}, Score: {score:.4f}")
    return

def test_lmf_predictions():
    print("Testing Logistic Matrix Factorization")
    test_data = get_test_data()
    test_lmf = LMF(interaction_matrix = test_data)
    test_lmf.load_from_cache()
    assert isinstance(test_lmf, LMF)
    movies = pd.read_csv("../data/ml-32m/movies.csv")
    test_user = 0
    print(f"Top 10 recommendations for user {test_user}: ")
    rec_indices, rec_scores = test_lmf.model.recommend(test_user, test_data, N=10, filter_already_liked_items=False)
    for idx, score in zip(rec_indices, rec_scores):
        movie_id = movies.iloc[idx+1]['movieId']
        movie_title = movies.iloc[idx]['title']
        print(f"MovieId: {movie_id}, Title: {movie_title}, Score: {score:.4f}")
    return

def test_ncf_predictions():
    print("Testing Neural Collaborative Filtering")
    
    # Load test data for reference
    test_data = get_test_data()
    
    num_users = 197270
    num_items = 42809
    embedding_dim = 32
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralCollaborativeFiltering(num_users, num_items, embedding_dim).to(device)
    model.load_from_cache()
    
    try:
        model.load_from_cache()
    except FileNotFoundError:
        print("Error: NCF model not found. Please train the model first.")
        return
        
    assert isinstance(model, NeuralCollaborativeFiltering)
    
    # Set model to evaluation mode
    model.eval()
    
    # Load movies data for displaying results
    movies = pd.read_csv("../data/ml-32m/movies.csv")
    
    # Test for a specific user
    test_user = 0  # Same as other tests
    print(f"Top 10 recommendations for user {test_user}: ")
    
    # Generate predictions for all movies
    with torch.no_grad():
        predictions = []
        
        # Process items in batches to avoid OOM errors
        batch_size = 1024
        for i in range(0, num_items, batch_size):
            batch_items = list(range(i, min(i + batch_size, num_items)))
            item_tensor = torch.tensor(batch_items, dtype=torch.long).to(device)
            user_tensor = torch.tensor([test_user] * len(batch_items), dtype=torch.long).to(device)
            
            batch_preds = model(user_tensor, item_tensor).cpu().numpy()
            
            for j, item_idx in enumerate(batch_items):
                predictions.append((item_idx, float(batch_preds[j])))
        
        # Sort predictions by score (descending)
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Get top 10
        top_predictions = predictions[:10]
        
        for idx, score in top_predictions:
            try:
                # Following the pattern used in test_bpr_predictions
                movie_id = movies.iloc[idx+1]['movieId']
                movie_title = movies.iloc[idx]['title']
                print(f"MovieId: {movie_id}, Title: {movie_title}, Score: {score:.4f}")
            except IndexError:
                print(f"Index error for movie index {idx}, Score: {score:.4f}")
    
    return

if __name__ == "__main__":
    test_ALS_predictions()
    # test_XGBoost_predictions()
    test_bpr_predictions()
    test_lmf_predictions()
    test_ncf_predictions()