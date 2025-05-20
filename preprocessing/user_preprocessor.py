import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class UserPreprocessor:
    """
    Preprocesses user data for feature engineering in the movie recommender system.
    This includes:
      - Aggregating user ratings
      - Extracting user-level genre features
      - Calculating user-level year statistics
      - Calculating user meta-tag statistics (religion, nsfw, oscar, etc.)
    """
    def __init__(self, ratings, movies, preprocessed_movies):
        self.ratings = ratings
        self.movies = movies
        self.preprocessed_movies = preprocessed_movies
        self.ratings = None
        self.movies = None
        self.preprocessed_movies = None
        self.user_features = None

    def aggregate_user_ratings(self):
        """Aggregates average rating and rating count for each user."""
        user_avg_ratings = self.ratings.groupby('userId')['rating'].mean()
        user_features = pd.DataFrame(user_avg_ratings)
        user_features.columns = ['avg_rating']

    def extract_user_genre_features(self):
        """Extracts user-level genre features such as watch count and average rating by genre."""
        # List of all genres the user has watched
        genres_usr_watched = self.ratings.merge(self.movies, on='movieId')[['userId', 'genres']]
        genres_usr_watched['genres'] = genres_usr_watched['genres'].str.split('|')
        # Get unique genres watched by each user
        genres_usr_watched = genres_usr_watched.groupby('userId')['genres'].sum().apply(lambda x: list(set(x)))
        self.user_features = self.user_features.join(genres_usr_watched.rename('unique_genres_watched'))
        self.user_features['unique_genres_watched'] = self.user_features['unique_genres_watched'].apply(lambda x: len(x) if isinstance(x, list) else 0)

        # 1) User watch count by genre
        # Explode genres so each row is (userId, genre)
        exploded = genres_usr_watched.explode('genres')
        # Remove any missing genres
        exploded = exploded[exploded['genres'].notnull()]
        # User watch count by genre (number of movies watched per genre)
        user_genre_watch_count = exploded.groupby(['userId', 'genres']).size().unstack(fill_value=0)

        # Join user watch count by genre
        self.user_features = self.user_features.join(user_genre_watch_count, rsuffix='_watch_count')


    def calculate_user_genre_avg_ratings(self):
        # 2) User average rating by genre
        # Merge ratings with movies to get genres for each rating
        ratings_with_genres = self.ratings.merge(self.movies[['movieId', 'genres']], on='movieId')
        ratings_with_genres['genres'] = ratings_with_genres['genres'].str.split('|')
        ratings_exploded = ratings_with_genres.explode('genres')
        ratings_exploded = ratings_exploded[ratings_exploded['genres'].notnull()]
        # Group by user and genre, then average the ratings
        user_genre_avg_rating = ratings_exploded.groupby(['userId', 'genres'])['rating'].mean().unstack()(fill_value=0)
        # Join user average rating by genre
        self.user_features = self.user_features.join(user_genre_avg_rating, rsuffix='_avg_rating')


    def calculate_user_year_stats(self):
        """Calculates mean, median, mode, and stddev of years of movies watched by each user."""
        def extract_year(title):
            """Extracts the year from a movie title string like 'Movie Title (1999)'. Returns np.nan if not found."""
            match = re.search(r'\((\d{4})\)', title)
            if match:
                return int(match.group(1))
            return np.nan

        # Add a 'year' column to the movies DataFrame
        self.movies['year'] = self.movies['title'].apply(extract_year)

        # Function to compute average year of movies watched by each user
        def year_stats_watched(movie_ids):
            years = self.movies[self.movies['movieId'].isin(movie_ids)]['year'].dropna()
            if len(years) == 0:
                return pd.Series([np.nan, np.nan, np.nan, np.nan], index=['year_mean', 'year_median', 'year_mode', 'year_stdDev'])
            mean = years.mean()
            median = years.median()
            stddev = years.std()
            mode = years.mode().iloc[0] if not years.mode().empty else np.nan
            return pd.Series([mean, median, mode, stddev], index=['year_mean', 'year_median', 'year_mode', 'year_stdDev'])
        # Compute year statistics for each user
        movies_usr_watched = self.ratings.groupby('userId')['movieId'].apply(list)
        user_year_stats = movies_usr_watched.apply(year_stats_watched)
        user_year_stats = user_year_stats.reset_index().rename(columns={"index": "userId"})
    
    def user_meta_stats(self, movie_ids, preprocessed_movies):
        """
        Returns the number and average score of religious, NSFW, and Oscar movies watched by a user.
        Output: (religious_count, religious_avg, nsfw_count, nsfw_avg, oscar_count, oscar_avg)
        """
        watched = preprocessed_movies[preprocessed_movies['movieId'].isin(movie_ids)]
        # Religious stats
        religious_movies = watched[watched['religion'] > 0]
        religious_count = len(religious_movies)
        religious_avg = religious_movies['religion'].mean() if religious_count > 0 else 0
        # NSFW stats
        nsfw_movies = watched[watched['nsfw'] > 0]
        nsfw_count = len(nsfw_movies)
        nsfw_avg = nsfw_movies['nsfw'].mean() if nsfw_count > 0 else 0
        # Oscar stats
        oscar_movies = watched[watched['oscars'] > 0]
        oscar_count = len(oscar_movies)
        oscar_avg = oscar_movies['oscars'].mean() if oscar_count > 0 else 0
        return religious_count, religious_avg, nsfw_count, nsfw_avg, oscar_count, oscar_avg

    def calculate_user_meta_tag_stats(self):
        """Calculates user-level meta-tag statistics (religion, nsfw, oscar, etc.)."""
        movies_usr_watched = self.ratings.groupby('userId')['movieId'].apply(list)
        meta_stats = movies_usr_watched['movieId'].apply(lambda x: self.user_meta_stats(x, self.preprocessed_movies))
        movies_usr_watched['religious_count'] = meta_stats.apply(lambda x: x[0])
        movies_usr_watched['religious_avg_score'] = meta_stats.apply(lambda x: x[1])
        movies_usr_watched['nsfw_count'] = meta_stats.apply(lambda x: x[2])
        movies_usr_watched['nsfw_avg_score'] = meta_stats.apply(lambda x: x[3])
        movies_usr_watched['oscar_count'] = meta_stats.apply(lambda x: x[4])
        movies_usr_watched['oscar_avg_score'] = meta_stats.apply(lambda x: x[5])
        self.user_features = self.user_features.join(movies_usr_watched.set_index('userId')[['religious_count', 'religious_avg_score', 'nsfw_count', 'nsfw_avg_score', 
                                                                           'oscar_count', 'oscar_avg_score']])

    def save_user_features(self, output_path):
        """Saves the final user features DataFrame to a CSV file."""
        pass
