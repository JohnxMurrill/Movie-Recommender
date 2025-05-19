import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from preprocessing import META_TAGS
import spacy

class MoviePreprocessor:
    """
    A class for preprocessing the movie data and creating embeddings for movies, incorporating tags and genres
    and meta information on tags.
    """
    def __init__(self, movies, ratings, tags):
        self.movies = movies
        self.ratings = ratings
        self.tags = tags
        self.processed_movies = None
        self.w2v_model = Word2Vec
        self.tfidf_vectorizer = None
        self.embedding_df = None
        self.meta_tags = META_TAGS
        self.nlp = spacy.load("en_core_web_md")

    def aggregate_ratings(self):
        """
        Compute the average rating for each movie and merge it into the movies DataFrame as a new column.
        """
        movie_avg_ratings = self.ratings.groupby('movieId')['rating'].mean()
        self.movies = self.movies.merge(movie_avg_ratings, on='movieId', how='left', suffixes=('', '_2'))
        return

    def clean_and_group_tags(self):
        """
        Lowercase all tags, group tags by movieId, join as comma-separated strings, and merge into the movies DataFrame.
        """
        self.tags['tag'] = self.tags['tag'].astype(str).str.lower()
        tags_by_movie = self.tags.groupby('movieId')['tag'].apply(lambda x: ','.join(x)).reset_index()
        self.movies = self.movies.merge(tags_by_movie, on='movieId', how='left', suffixes=('', '_2'))
        pass

    def prune_tags(self):
        """
        Remove tags that appear less than or equal to 50 times in the tags DataFrame.
        """
        tag_counts = self.tags['tag'].value_counts()
        mask = self.tags['tag'].map(tag_counts) > 50
        self.tags = self.tags[mask]
        pass

    def deduplicate_tags(self):
        """
        Remove duplicate and empty tags for each movie, preserving order, and update the movies DataFrame.
        """
        def dedup_tags(tag_str):
            tags = tag_str.split(',')
            # Remove empty strings and deduplicate while preserving order
            seen = set()
            deduped = [t for t in tags if t and not (t in seen or seen.add(t))]
            return ','.join(deduped)

        self.movies['tag'] = self.movies['tag'].fillna('').apply(dedup_tags)

    def extract_meta_tags(self):
        """
        Extract and score meta-tags (NSFW, religious, childrens) for each movie and add them as columns in the movies DataFrame.
        """
        def score_religion_nsfw(tags_str):
            tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            nsfw = sum(1 for tag in tags if tag in self.meta_tags['nsfw'])
            religion = sum(1 for tag in tags if tag in self.meta_tags['religious'])
            childrens = sum(1 for tag in tags if tag in self.meta_tags['childrens'])
            religion_norm = religion / len(self.meta_tags['religious']) if len(self.meta_tags['religious']) > 0 else 0
            return nsfw, religion_norm, childrens
        self.movies[['nsfw', 'religion', 'childrens']] = self.movies['tag'].fillna('').apply(lambda x: pd.Series(score_religion_nsfw(x)))
        self.movies['nsfw'] = self.movies['nsfw'].astype(int)
        self.movies['religion'] = self.movies['religion'].astype(float)

    def extract_actors(self):
        """
        Use spaCy NER to identify and extract actor/person tags from the tag list for each movie and add as a new column.
        """
        people = {}
        def is_person(tag):
            if tag in people:
                return people[tag]
            doc = self.nlp(tag)
            person = any(ent.label_ == "PERSON" for ent in doc.ents)
            people[tag] = person
            return person
        self.movies['actors'] = self.movies['tag'].apply(lambda x: [tag for tag in x.split(',') if is_person(tag)])

    def extract_oscars(self):
        """
        Extract Oscar wins/nominations from tags, remove them from the tag list, and add an 'oscars' column to the movies DataFrame.
        """
        def extract_oscars(tags_str):
            tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            oscars = 0.0
            new_tags = []
            for tag in tags:
                if 'oscar winner' in tag or 'oscar (best' in tag or 'oscar winner:' in tag:
                    oscars += 2.0
                elif 'oscar nominee' in tag:
                    oscars += 0.5
                else:
                    new_tags.append(tag)
            return ','.join(new_tags), oscars
        self.movies[['tag', 'oscars']] = self.movies['tag'].fillna('').apply(lambda x: pd.Series(extract_oscars(x)))
        self.movies['oscars'] = self.movies['oscars'].astype(float)

    def remove_genre_from_tags(self):
        """
        Remove tags that are exactly equal to a genre from the tag list for each movie.
        """
        def drop_genre_tags(tags_str, genres_list):
            tags = [t.strip() for t in tags_str.split(',') if t.strip()]
            filtered_tags = [tag for tag in tags if tag not in genres_list]
            return ','.join(filtered_tags)  
        # Apply the function to the movies DataFrame
        genres_list = self.meta_tags['genres']
        self.movies['tag'] = self.movies['tag'].fillna('').apply(lambda x: drop_genre_tags(x, genres_list))
    
    def embed_genres(self):
        """
        One-hot encode the genres column and drop the original genres and '(no genres listed)' columns from the movies DataFrame.
        """
        genre_dummies = self.movies['genres'].str.get_dummies(sep='|')
        self.movies = pd.concat([self.movies, genre_dummies], axis=1)
        self.movies = self.movies.drop(columns=['genres', "(no genres listed)"])

    def vectorize_tags(self):
        """
        Prepare tag lists for Word2Vec and TF-IDF, train a Word2Vec model, and fit a TfidfVectorizer on the tag lists.
        """
        # Prepare sentences for Word2Vec (list of tags per movie)
        sentences = self.movies['tag'].fillna('').apply(lambda x: [tag for tag in x.split(',') if tag]).tolist()
        # Train Word2Vec model
        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=150,   # Embedding dimension
            window=5,          # Context window
            min_count=2,       # Ignore tags that appear < 2 times
            workers=4          # Parallel processing
        )
        # Prepare tag strings for TF-IDF (space-separated tags per movie)
        tag_strings = self.movies['tag'].fillna('').apply(lambda x: ' '.join([tag for tag in x.split(',') if tag]))
        # Fit TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(tag_strings)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out()

    def compute_tag_embeddings(self):
        """
        Compute TF-IDF-weighted average tag embeddings for each movie using the trained Word2Vec and TF-IDF models, and merge the embeddings into the movies DataFrame.
        """
        movie_embeddings = {}
        for idx, movie_id in enumerate(self.movies['movieId']):
            # Get TF-IDF vector for this movie
            tfidf_vector = self.tfidf_matrix[idx].toarray().flatten()
            # Get the tags for this movie
            movie_tags = [tag for tag in self.movies.iloc[idx]['tag'].split(',') if tag]
            # Initialize embedding vector
            weighted_embedding = np.zeros(self.w2v_model.vector_size)
            total_weight = 0
            # For each tag, add its weighted embedding
            for tag in movie_tags:
                if tag in self.w2v_model.wv and tag in self.feature_names:
                    tfidf_idx = np.where(self.feature_names == tag)[0]
                    if len(tfidf_idx) > 0:
                        weight = tfidf_vector[tfidf_idx[0]]
                        weighted_embedding += self.w2v_model.wv[tag] * weight
                        total_weight += weight
            # Normalize by total weight if non-zero
            if total_weight > 0:
                weighted_embedding /= total_weight
            movie_embeddings[movie_id] = weighted_embedding
        # Convert to DataFrame for easier use
        self.embedding_df = pd.DataFrame.from_dict(movie_embeddings, orient='index')
        self.embedding_df.index.name = 'movieId'
        self.embedding_df.columns = [f'tag_embedding_{i}' for i in range(self.w2v_model.vector_size)]
        # Join with the original movies dataframe
        self.movies = self.movies.merge(self.embedding_df, on='movieId')
    
    def drop_unnecessary_columns(self):
        """
        Drop unnecessary columns from the movies DataFrame.
        As of the first version, this is all remaining text columns and the movieId since initial
        models don't use text.
        """
        self.movies = self.movies.drop(columns=['tag', 'title'])

    def save_processed_data(self, output_path):
        """
        Save the processed movies DataFrame to a CSV file at the specified output path.
        """
        self.movies.to_csv(output_path, index=False)
