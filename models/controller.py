from concurrent.futures import ThreadPoolExecutor
from bayesian_personalized_rankings.bpr import BPR
from logistic_matrix_factorization.lmf import LMF
from als.als import AlternatingLeastSquares
from scipy import sparse
import pandas as pd
import numpy as np
from collections import Counter

# Orchestration Service
def get_blended_prediction(user_id, item_features, movies):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(BayesPR.model.recommend, userid=user_id, user_items=item_features[user_id], N=100, filter_already_liked_items=False),
            executor.submit(LogMF.model.recommend, userid=user_id, user_items=item_features[user_id], N=100, filter_already_liked_items=False),
            executor.submit(ALS.model.recommend, user_id, item_features[user_id], N=100, filter_already_liked_items=False),
        ]
        results = [future.result() for future in futures]
    results = blend_ranked_lists(results, movies['movieId'], top_n=20)
    
    print(f"Top 20 recommendations for user {user_id}:")
    for idx, votes, avg_rank in results:
        movie_id = movies.iloc[idx]['movieId']
        movie_title = movies.iloc[idx]['title']
        print(f"MovieId: {movie_id}, Title: {movie_title}, Votes: {votes}, AvgRank: {avg_rank:.4f}")
    return

def blend_ranked_lists(list_of_lists, movie_ids, top_n=20):
    """
    Blends multiple ranked lists into a single ranked list using voting.
    Each model's recommendation is treated as a vote for the movie.
    """
    # Flatten the list of lists and count votes
    all_recommendations = []
    for model_results, scores in list_of_lists:
        all_recommendations.extend(model_results)

    # Count votes for each movie
    vote_counts = Counter()
    for idx in all_recommendations:
        vote_counts[idx] += 1

    # Sort by votes and average rank
    sorted_movies = sorted(vote_counts.items(), key=lambda x: (-x[1], x[0]))

    # Get top N recommendations
    top_recommendations = [(idx, votes, np.mean([rank for rank in all_recommendations if rank == idx])) for idx, votes in sorted_movies[:top_n]]
    
    return top_recommendations

if __name__ == '__main__':
    data = sparse.load_npz("../data/processed_data/user_movie_interactions.npz")
    movies = pd.read_csv("../data/ml-32m/movies.csv")
    ALS = AlternatingLeastSquares(data)
    ALS.load_from_cache()
    BayesPR = BPR(data)
    BayesPR.load_from_cache()
    LogMF = LMF(data)
    LogMF.load_from_cache()
    get_blended_prediction(0, data, movies)