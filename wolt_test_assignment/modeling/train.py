from pathlib import Path
import typer
from loguru import logger
import numpy as np
import joblib

from models.lstm import create_regressor
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from wolt_test_assignment.config import MODELS_DIR, PROCESSED_DATA_DIR


def load_scaled_data():
    """
    Load the scaled training and test datasets, along with the scaler.

    Returns:
        tuple: 
            - training_set_scaled (numpy.ndarray): Scaled training set features.
            - test_set_scaled (numpy.ndarray): Scaled test set features.
            - scaler (object): Fitted scaler for inverse transformations.
    """
    training_set_scaled = np.load(PROCESSED_DATA_DIR / "training_feature_set_scaled.npy")
    test_set_scaled = np.load(PROCESSED_DATA_DIR / "test_feature_set_scaled.npy")
    scaler = joblib.load(PROCESSED_DATA_DIR / "scaler.pkl")

    return training_set_scaled, test_set_scaled, scaler


def train_linear_regression(training_days: int, n_steps: int):
    """
    Train a linear regression model for time series forecasting.

    Parameters:
        training_days (int): Number of past days to use as input features.
        n_steps (int): Number of future steps to predict.

    Returns:
        object: Trained linear regression or multi-output regression model.
    """
    logger.info("Loading and preprocessing data...")
    training_set_scaled, test_set_scaled, scaler = load_scaled_data()

    X_train, y_train = [], []

    for i in range(training_days, len(training_set_scaled) - n_steps + 1):
        X_train.append(training_set_scaled[i - training_days:i])
        y_train.append(training_set_scaled[i:i + n_steps, 0])

    X_train = np.array(X_train).reshape((len(X_train), -1))  # Flatten for regression
    y_train = np.array(y_train)

    logger.info("Training the linear regression model...")
    if n_steps == 1:
        y_train = y_train.flatten()
        model = LinearRegression()
    else:
        model = MultiOutputRegressor(LinearRegression())

    model.fit(X_train, y_train)
    return model


def train_RNN(epochs: int, batch_size: int, training_days: int, n_steps: int):
    """
    Train an LSTM model for time series forecasting.

    Parameters:
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        training_days (int): Number of past days to use as input features.
        n_steps (int): Number of future steps to predict.

    Returns:
        Sequential: Trained LSTM model.
    """
    logger.info("Loading and preprocessing data...")
    training_set_scaled, test_set_scaled, scaler = load_scaled_data()

    X_train, y_train = [], []

    for i in range(training_days, len(training_set_scaled) - n_steps + 1):
        X_train.append(training_set_scaled[i - training_days:i])
        y_train.append(training_set_scaled[i:i + n_steps, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    input_shape = (X_train.shape[1], X_train.shape[2])

    logger.info("Building the LSTM model...")
    regressor = create_regressor(input_shape, n_steps)

    logger.info("Training the LSTM model...")
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    return regressor


app = typer.Typer()

@app.command()
def main(
    epochs: int = 100,
    batch_size: int = 32,
    training_days: int = 40,
    n_steps: int = 20
):
    """
    Main function to train and save both LSTM and Linear Regression models.

    Parameters:
        epochs (int): Number of epochs for training the LSTM.
        batch_size (int): Batch size for training the LSTM.
        training_days (int): Number of past days to use for prediction.
        n_steps (int): Number of future steps to predict.
    """
    logger.info("Training the LSTM model...")
    lstm_regressor = train_RNN(epochs, batch_size, training_days, n_steps)
    lstm_model_name = f"model_LSTM_featureDays_{training_days}_steps_{n_steps}.h5"
    lstm_model_path = MODELS_DIR / lstm_model_name
    lstm_regressor.save(lstm_model_path)
    logger.success("LSTM model training complete and saved.")

    logger.info("Training the Linear Regression model...")
    lr_regressor = train_linear_regression(training_days, n_steps)
    lr_model_name = f"model_LR_featureDays_{training_days}_steps_{n_steps}.joblib"
    lr_model_path = MODELS_DIR / lr_model_name
    joblib.dump(lr_regressor, lr_model_path)
    logger.success("Linear Regression model training complete and saved.")


if __name__ == "__main__":
    app()
