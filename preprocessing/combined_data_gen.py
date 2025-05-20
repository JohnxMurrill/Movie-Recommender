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
    user_features = user_df.values
    movie_features = movie_df.values
    user_ids = user_df.index.values
    movie_ids = movie_df.index.values
    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    print("Loading user-movie interaction matrix (sparse)...")
    interaction_data = np.load(INTERACTIONS_PATH)
    if 'arr_0' in interaction_data:
        interaction_matrix = sparse.coo_matrix(interaction_data['arr_0'])
    else:
        # If saved as separate arrays
        interaction_matrix = sparse.coo_matrix((interaction_data['data'], (interaction_data['row'], interaction_data['col'])),
                                              shape=interaction_data['shape'])

    n_interactions = interaction_matrix.nnz
    print(f"Total interactions: {n_interactions}")

    # Prepare for batch writing
    writer = None
    batch_rows = []
    batch_count = 0
    for idx, (u, m, r) in enumerate(zip(interaction_matrix.row, interaction_matrix.col, interaction_matrix.data)):
        # Lookup features
        user_vec = user_features[u]
        movie_vec = movie_features[m]
        row = np.concatenate([user_vec, movie_vec, [r]])
        batch_rows.append(row)
        if len(batch_rows) >= BATCH_SIZE or idx == n_interactions - 1:
            batch_df = pd.DataFrame(batch_rows)
            # Write batch to Parquet
            table = pa.Table.from_pandas(batch_df)
            if writer is None:
                writer = pq.ParquetWriter(OUTPUT_PARQUET_PATH, table.schema)
            writer.write_table(table)
            print(f"Wrote batch {batch_count+1} ({len(batch_rows)} rows)")
            batch_rows = []
            batch_count += 1
    if writer:
        writer.close()
    print(f"Done. Output saved to {OUTPUT_PARQUET_PATH}")

if __name__ == "__main__":
    main()
