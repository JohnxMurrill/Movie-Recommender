import numpy as np
import pandas as pd
import os
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

# CONFIGURABLES
USER_FEATURES_PATH = '../pre_processed_users.csv'
MOVIE_FEATURES_PATH = '../pre_processed_movies.csv'
INTERACTIONS_PATH = '../data/ml-32m/ratings.csv'
OUTPUT_PARQUET_PATH = '../xgboost_training_data.parquet'
BATCH_SIZE = 500_000  # Adjust as needed for your memory


def main():
    print("Loading user and movie features into memory...")
    user_df = pd.read_csv(USER_FEATURES_PATH, index_col=0)
    movie_df = pd.read_csv(MOVIE_FEATURES_PATH, index_col=0)
    user_ids = user_df.index.values
    movie_ids = movie_df.index.values
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    print("Reading ratings.csv in chunks...")
    reader = pd.read_csv(INTERACTIONS_PATH, usecols=["userId", "movieId", "rating"], chunksize=BATCH_SIZE)
    writer = None
    batch_count = 0
    for chunk in reader:
        batch_rows = []
        for row in chunk.itertuples(index=False):
            userId, movieId, rating = row
            if userId in user_id_to_idx and movieId in movie_id_to_idx:
                user_vec = user_df.loc[userId].values
                movie_vec = movie_df.loc[movieId].values
                # For regression, use rating; for binary classification, use e.g. int(rating >= 4.0)
                label = rating  # or: int(rating >= 4.0)
                batch_rows.append(np.concatenate([user_vec, movie_vec, [label]]))
        if batch_rows:
            batch_df = pd.DataFrame(batch_rows)
            table = pa.Table.from_pandas(batch_df)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PARQUET_PATH, table.schema)
            writer.write_table(table)
            print(f"Wrote batch {batch_count+1} ({len(batch_rows)} rows)")
            batch_count += 1
    if writer:
        writer.close()
    print(f"Done. Output saved to {OUTPUT_PARQUET_PATH}")

if __name__ == "__main__":
    main()
