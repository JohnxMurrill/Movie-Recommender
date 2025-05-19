import pandas as pd
import numpy as np
from collections import Counter
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from sklearn.cluster import DBSCAN

# Load the data
tags_df = pd.read_csv('tags.csv')
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Basic exploration
print(f"Number of unique tags: {tags_df['tag'].nunique()}")
print(f"Number of unique movies with tags: {tags_df['movieId'].nunique()}")
print(f"Number of unique users who tagged: {tags_df['userId'].nunique()}")

# 1. Clean and normalize tags
def clean_tag(tag):
    if not isinstance(tag, str):
        return ""
    # Convert to lowercase
    tag = tag.lower().strip()
    # Remove special characters
    tag = re.sub(r'[^\w\s]', '', tag)
    # Remove numbers
    tag = re.sub(r'\d+', '', tag)
    # Remove extra spaces
    tag = re.sub(r'\s+', ' ', tag).strip()
    return tag

# Apply cleaning
tags_df['clean_tag'] = tags_df['tag'].apply(clean_tag)
tags_df = tags_df[tags_df['clean_tag'] != ""]  # Remove empty tags

# 2. Tag frequency analysis
tag_counts = Counter(tags_df['clean_tag'])
print(f"Top 20 most common tags: {tag_counts.most_common(20)}")

# Visualize tag frequency distribution
plt.figure(figsize=(12, 6))
tag_freq = pd.Series(tag_counts).value_counts().sort_index()
sns.histplot(tag_freq, bins=50)
plt.xlabel('Tag Occurrence Count')
plt.ylabel('Number of Tags')
plt.title('Distribution of Tag Frequencies')
plt.xscale('log')
plt.savefig('tag_frequency_distribution.png')

# 3. Prune infrequent tags
min_tag_count = 5  # Minimum number of occurrences to keep a tag
frequent_tags = {tag for tag, count in tag_counts.items() if count >= min_tag_count}
tags_df = tags_df[tags_df['clean_tag'].isin(frequent_tags)]
print(f"Number of tags after pruning infrequent ones: {tags_df['clean_tag'].nunique()}")

# 4. Group tags by movie for similarity analysis
movie_tags = {}
for movie_id, group in tags_df.groupby('movieId'):
    movie_tags[movie_id] = list(group['clean_tag'])

# 5. Create tag embeddings using Word2Vec
# Convert to format for Word2Vec
tag_sentences = list(movie_tags.values())
# Train Word2Vec model
w2v_model = Word2Vec(sentences=tag_sentences, vector_size=100, window=5, min_count=1, workers=4)

# 6. Find similar tags using embeddings
def get_similar_tags(tag, model, n=10):
    if tag in model.wv:
        similar_tags = model.wv.most_similar(tag, topn=n)
        return similar_tags
    return []

# Example: Get similar tags for a common tag
common_tag = tag_counts.most_common(1)[0][0]
print(f"Tags similar to '{common_tag}':")
for tag, similarity in get_similar_tags(common_tag, w2v_model):
    print(f"  {tag}: {similarity:.4f}")

# 7. Cluster similar tags
# Extract vectors for all tags
tag_vectors = {}
for tag in frequent_tags:
    if tag in w2v_model.wv:
        tag_vectors[tag] = w2v_model.wv[tag]

# Convert to array for clustering
tags_to_cluster = list(tag_vectors.keys())
vectors = np.array([tag_vectors[tag] for tag in tags_to_cluster])

# Apply DBSCAN clustering
clustering = DBSCAN(eps=0.5, min_samples=3).fit(vectors)
tag_clusters = {tag: cluster for tag, cluster in zip(tags_to_cluster, clustering.labels_)}

# Create mapping from original tags to cluster representatives
cluster_to_tags = {}
for tag, cluster in tag_clusters.items():
    if cluster != -1:  # -1 means noise in DBSCAN
        if cluster not in cluster_to_tags:
            cluster_to_tags[cluster] = []
        cluster_to_tags[cluster].append(tag)

# For each cluster, select the most frequent tag as representative
cluster_representatives = {}
for cluster, cluster_tags in cluster_to_tags.items():
    # Get frequency of each tag in the cluster
    tag_freqs = {tag: tag_counts[tag] for tag in cluster_tags}
    # Select the most frequent one as representative
    representative = max(tag_freqs.items(), key=lambda x: x[1])[0]
    cluster_representatives[cluster] = representative
    print(f"Cluster {cluster}: {cluster_tags} → {representative}")

# 8. Replace similar tags with their cluster representatives
def get_representative_tag(tag):
    if tag in tag_clusters and tag_clusters[tag] != -1:
        return cluster_representatives[tag_clusters[tag]]
    return tag

tags_df['representative_tag'] = tags_df['clean_tag'].apply(get_representative_tag)
print(f"Number of unique tags after clustering: {tags_df['representative_tag'].nunique()}")

# 9. Create movie-tag matrix using TF-IDF
# Group tags by movie after clustering
movie_tags_clustered = {}
for movie_id, group in tags_df.groupby('movieId'):
    movie_tags_clustered[movie_id] = list(group['representative_tag'])

# Convert to format for TF-IDF
movie_tag_documents = {}
for movie_id, tags in movie_tags_clustered.items():
    movie_tag_documents[movie_id] = ' '.join(tags)

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
movie_ids = list(movie_tag_documents.keys())
movie_tag_docs = [movie_tag_documents[movie_id] for movie_id in movie_ids]
tfidf_matrix = vectorizer.fit_transform(movie_tag_docs)

# Create DataFrame for easier analysis
tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    index=movie_ids,
    columns=vectorizer.get_feature_names_out()
)

print(f"Shape of TF-IDF matrix: {tfidf_matrix.shape}")

# 10. Save processed data for recommendation system
tags_processed = tags_df[['userId', 'movieId', 'representative_tag', 'timestamp']]
tags_processed.to_csv('processed_tags.csv', index=False)
tfidf_df.to_csv('movie_tag_tfidf.csv')

# 11. Example: Function to get tag-based movie similarity
def get_similar_movies_by_tags(movie_id, tfidf_matrix, movie_ids, n=10):
    movie_index = movie_ids.index(movie_id)
    movie_vector = tfidf_matrix[movie_index:movie_index+1]
    
    # Calculate similarity
    sim_scores = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    # Get top N similar movies (excluding the movie itself)
    similar_indices = sim_scores.argsort()[:-n-2:-1]
    similar_movies = [(movie_ids[i], sim_scores[i]) for i in similar_indices if i != movie_index]
    
    return similar_movies

# Example usage:
example_movie_id = movie_ids[0]
movie_title = movies_df[movies_df['movieId'] == example_movie_id]['title'].values[0]
print(f"\nMovies similar to '{movie_title}' based on tags:")
similar_movies = get_similar_movies_by_tags(example_movie_id, tfidf_matrix, movie_ids)
for similar_id, similarity in similar_movies:
    similar_title = movies_df[movies_df['movieId'] == similar_id]['title'].values[0]
    print(f"  {similar_title}: {similarity:.4f}")