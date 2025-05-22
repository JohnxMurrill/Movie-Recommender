import numpy as np
import pandas as pd
from scipy import sparse
from models.als.als import AlternatingLeastSquares
from models.collaborative_filtering.user_user import UserUserCollaborativeFiltering
from models.content_filtering.item_item import ItemItemContentFiltering
from models.bayesian_personalized_rankings.bpr import BPR
from models.logistic_matrix_factorization.lmf import LMF

def get_test_data():
    # Test user-movie interactions
    test_user_movie_interactions = sparse.load_npz("../data/processed_data/user_movie_interactions.npz")
    return test_user_movie_interactions

def test_CF_predictions():
    pass
    


def test_Content_predictions():
    print("Testing Content-based filtering")
    test_data = get_test_data()
    test_CF = ItemItemContentFiltering(test_data)
    test_CF.load_from_cache()
    assert isinstance(test_CF, ItemItemContentFiltering)
    movies = pd.read_csv("../data/ml-32m/movies.csv")
    test_user = 0
    test_item = 2353 # 2353 - 4.5, 2005 - 1.0, 1198 - 3.5
    print("User, item: ", test_user, test_item)
    print(f"Top 10 recommendations for user {test_user}: ")
    rec = test_CF.recommend(test_user)
    for idx, score in rec:
        movie_id = movies.iloc[idx]['movieId']
        movie_title = movies.iloc[idx]['title']
        print(f"MovieId: {movie_id}, Title: {movie_title}, Score: {score:.4f}")

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

if __name__ == "__main__":
    test_ALS_predictions()
    # test_XGBoost_predictions()
    test_CF_predictions()
    # test_Content_predictions()
    test_bpr_predictions()
    test_lmf_predictions()