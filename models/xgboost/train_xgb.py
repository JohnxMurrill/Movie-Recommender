import os
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import xgboost as xgb
from models import logger
import matplotlib.pyplot as plt


# CONFIGURABLES
PARQUET_PATH = 'data/processed_data/xgboost_training_data_cleaned.parquet'
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/xgb_model_gpu.json')
BATCH_SIZE = 1_000_000  # For local testing, reduce to e.g. 100_000
N_ESTIMATORS = 100
MAX_DEPTH = 8
LEARNING_RATE = 0.2
EARLY_STOPPING_ROUNDS = 7
USE_GPU = True  # Set to False to use CPU
SAMPLE_ROWS = 20_000_000  # For local testing, set low; for EC2, set to None for full data
CHECKPOINT_DIR = 'xgb_checkpoints/'
CHECKPOINT_INTERVAL = 10  # Save every N boosting rounds

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def get_xgb_params():
    params = {
        'tree_method': 'hist',
        'device': 'cuda' if USE_GPU else 'cpu',
        'n_estimators': N_ESTIMATORS,
        'max_depth': MAX_DEPTH,
        'learning_rate': LEARNING_RATE,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'n_jobs': -1,
    }
    return params

def batch_generator(parquet_path, batch_size, sample_rows=None):
    parquet_file = pq.ParquetFile(parquet_path)
    rows_read = 0
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        df = pa.Table.from_batches([batch]).to_pandas()
        if sample_rows is not None and rows_read + len(df) > sample_rows:
            df = df.iloc[:sample_rows - rows_read]
        X = df.drop(df.columns[-1], axis=1).values.astype(np.float32)
        y = df[df.columns[-1]].values.astype(np.float32)
        yield X, y
        rows_read += len(df)
        if sample_rows is not None and rows_read >= sample_rows:
            break

def main():
    logger.info(f"Loading data from {PARQUET_PATH} in batches...")
    params = get_xgb_params()
    evals_result = {}
    # For demonstration, use the first batch as validation
    gen = batch_generator(PARQUET_PATH, BATCH_SIZE, sample_rows=SAMPLE_ROWS)
    X_train, y_train = next(gen)
    try:
        X_val, y_val = next(gen)
    except StopIteration:
        X_val, y_val = X_train, y_train  # If only one batch, use same for val
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    logger.info(f"Training XGBoost with GPU={'gpu_hist' if USE_GPU else 'hist'}...")
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=N_ESTIMATORS,
        evals=[(dtrain, 'train'), (dval, 'val')],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        callbacks=[xgb.callback.TrainingCheckPoint(directory=CHECKPOINT_DIR, name='xgb', interval=CHECKPOINT_INTERVAL)]
    )
    logger.info(f"Saving model to {MODEL_PATH}")
    booster.save_model(MODEL_PATH)
    logger.info("Training complete. Best iteration: %s", booster.best_iteration)
    logger.info("Eval results: %s", evals_result)

    # Uncomment to view feature importance
    # Get feature importance by gain
    # importance = booster.get_score(importance_type='gain')
    # logger.info("Feature importance by gain: %s", importance)

    # Plot feature importance
    # xgb.plot_importance(booster, importance_type='gain', max_num_features=20)
    # plt.show()

if __name__ == "__main__":
    main()
