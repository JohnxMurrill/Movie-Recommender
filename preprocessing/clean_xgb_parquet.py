import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import ast

INPUT_PARQUET = '../data/processed_data/xgboost_training_data.parquet'
OUTPUT_PARQUET = '../data/processed_data/xgboost_training_data_cleaned.parquet'
COLUMNS_TO_DROP = ['userId', 'movieId', 'title']  # Drop these columns
ACTORS_COLUMN = 'actors'  # Tokenize this column if present
BATCH_SIZE = 500_000  # Adjust as needed for your memory


def tokenize_actors(actors_str):
    if pd.isnull(actors_str):
        return []
    try:
        if isinstance(actors_str, str):
            actors = ast.literal_eval(actors_str)
            if isinstance(actors, list):
                return [str(a).strip() for a in actors]
            return [a.strip() for a in actors_str.split(',')]
        return []
    except Exception:
        return [a.strip() for a in str(actors_str).split(',')]


def main():
    print(f"Opening {INPUT_PARQUET} for batch cleaning...")
    parquet_file = pq.ParquetFile(INPUT_PARQUET)
    writer = None
    total_rows = 0
    for batch_idx, batch in enumerate(parquet_file.iter_batches(batch_size=BATCH_SIZE)):
        df = pa.Table.from_batches([batch]).to_pandas()
        # Print columns 0-52 for the first row in the batch for inspection
        # print(f"Batch {batch_idx+1} - Columns 0-52 sample: {df.iloc[0, 53:].tolist()}")
        # Drop columns with dtype 'object' (string columns)
        string_cols = df.select_dtypes(include=['object']).columns
        if len(string_cols) > 0:
            df = df.drop(columns=string_cols)
        # Write batch to output parquet
        table_cleaned = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(OUTPUT_PARQUET, table_cleaned.schema)
            # Save schema to file on first batch
            with open(OUTPUT_PARQUET + '.schema.txt', 'w', encoding='utf-8') as f:
                f.write(str(table_cleaned.schema))
        writer.write_table(table_cleaned)
        total_rows += len(df)
        print(f"Processed batch {batch_idx+1}, rows written: {total_rows}")
    if writer:
        writer.close()
    print(f"Done. Cleaned data saved to {OUTPUT_PARQUET}")

if __name__ == "__main__":
    main()
