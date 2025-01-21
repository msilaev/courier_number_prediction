from pathlib import Path
import typer
from loguru import logger
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from wolt_test_assignment.config import PROCESSED_DATA_DIR, INTERIM_DATA_DIR, SPLIT_DATE

def prepare_features(df, split_date):
    """
    Prepare the training and test datasets by scaling selected features and splitting the data.

    Parameters:
    df (DataFrame): The preprocessed dataset containing features and target columns.
    split_date (str): The date (in string format) to split the dataset into training and test sets.

    Saves:
    - Scaled total dataset (`total_feature_set_scaled.npy`).
    - Scaled training dataset (`training_feature_set_scaled.npy`).
    - Scaled test dataset (`test_feature_set_scaled.npy`).
    - Test target set (`test_target_set.npy`).
    - Scaler object (`scaler.pkl`).
    """
    features = ['courier_partners_online', 'precipitation'] + [col for col in df.columns if col.startswith('day_of_week')]
    total_set = df[features].values
    training_set = df[features][df['date'] < split_date].values
    test_set = df[features][df['date'] >= split_date].values

    scaler = StandardScaler()
    scaler.fit(total_set)
    total_set_scaled = scaler.transform(total_set)
    training_set_scaled = scaler.transform(training_set)
    test_set_scaled = scaler.transform(test_set)

    # Save the scaled datasets and scaler
    np.save(PROCESSED_DATA_DIR / "total_feature_set_scaled.npy", total_set_scaled)
    np.save(PROCESSED_DATA_DIR / "training_feature_set_scaled.npy", training_set_scaled)
    np.save(PROCESSED_DATA_DIR / "test_feature_set_scaled.npy", test_set_scaled)
    np.save(PROCESSED_DATA_DIR / "test_target_set.npy", df['courier_partners_online'][df['date'] >= split_date].values)
    joblib.dump(scaler, PROCESSED_DATA_DIR / "scaler.pkl")

def create_sequences(data, training_days, n_steps):
    """
    Create input-output sequences for time series forecasting.

    Parameters:
    data (array): The scaled dataset containing features.
    training_days (int): The number of past days to use as input for prediction.
    n_steps (int): The number of future steps to predict.

    Returns:
    tuple: 
    - X (array): Array of input sequences of shape `(num_samples, training_days, num_features)`.
    - y (array): Array of target sequences of shape `(num_samples, n_steps)`.
    """
    X, y = [], []
    for i in range(training_days, len(data) - n_steps + 1):
        X.append(data[i - training_days:i])
        y.append(data[i:i + n_steps, 0])  # Predict 'n_steps' future values of 'courier_partners_online'

    return np.array(X), np.array(y)

app = typer.Typer()

@app.command()
def main(input_path: Path = INTERIM_DATA_DIR / "daily_cp_activity_dataset.csv"):
    """
    Main entry point for generating features from the input dataset.

    Parameters:
    input_path (Path): The file path to the input dataset. Defaults to 
                       `INTERIM_DATA_DIR / "daily_cp_activity_dataset.csv"`.

    Saves:
    - Preprocessed feature sets and scaler for model training and testing.
    """
    logger.info("Generating features from dataset...")
    df = pd.read_csv(input_path)
    prepare_features(df, SPLIT_DATE)
    logger.success("Feature generation complete.")

if __name__ == "__main__":
    app()
