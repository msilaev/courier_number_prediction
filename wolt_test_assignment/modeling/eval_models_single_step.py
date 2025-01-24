import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path

# Related third-party imports
import absl.logging
import numpy as np
import typer
from joblib import load
from keras.models import load_model
from loguru import logger

# Local application/library-specific imports
from wolt_test_assignment.config import FIGURES_DIR, MODELS_DIR, SPLIT_DATE
from wolt_test_assignment.modeling.utils import calculate_metrics, load_features_target
from wolt_test_assignment.plots import plot_prediction

# Suppress TensorFlow INFO and WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress `absl` logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Suppress specific Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Compiled the loaded model.*")


def eval_lstm_model(
    model_path: Path = MODELS_DIR / "model_RNN_next_day.h5",
    training_days: int = 40,
    n_steps: int = 1,
):
    """
    Evaluate the LSTM model on test data.

    Parameters:
        model_path (Path): Path to the saved LSTM model file.
        training_days (int): Number of past days to use as input for predictions.
        n_steps (int): Number of future steps to predict.

    Returns:
        tuple:
            - y_test_original (np.ndarray): Actual target values from the test set.
            - predicted_courier_number_original (np.ndarray): Predicted values
            - transformed back to the original scale.
            - start_date_str (str): Start date of predictions in "YYYY-MM-DD" format.
    """
    test_target, _, test_set_scaled, scaler = load_features_target()

    x_test = []
    y_test_original = []
    test_target = test_target.reshape(-1, 1)

    for i in range(training_days, len(test_set_scaled) - n_steps + 1):
        x_test.append(test_set_scaled[i - training_days : i])
        y_test_original.append(test_target[i, 0])

    x_test = np.array(x_test)
    y_test_original = np.array(y_test_original)

    model = load_model(model_path)
    predicted_courier_number = model.predict(x_test, verbose=0)

    predicted_full = np.zeros((predicted_courier_number.shape[0], test_set_scaled.shape[1]))
    predicted_full[:, 0] = predicted_courier_number.flatten()
    predicted_original = scaler.inverse_transform(predicted_full)
    predicted_courier_number_original = predicted_original[:, 0]

    split_date_dt = datetime.strptime(SPLIT_DATE, "%Y-%m-%d")
    start_date = split_date_dt + timedelta(training_days)
    start_date_str = start_date.strftime("%Y-%m-%d")

    integer_predicted_courier_number = np.round(predicted_courier_number_original[training_days:])

    return (
        y_test_original[training_days:],
        integer_predicted_courier_number,
        start_date_str,
    )


def eval_lr_model(
    model_path: Path = MODELS_DIR / "model_LR.joblib",
    training_days: int = 40,
    n_steps: int = 1,
    split_date: str = SPLIT_DATE,
):
    """
    Evaluate the Linear Regression model on test data.

    Parameters:
        model_path (Path): Path to the saved Linear Regression model file.
        training_days (int): Number of past days to use as input for predictions.
        n_steps (int): Number of future steps to predict.
        split_date (str): The date to split the dataset into training and test sets.

    Returns:
        tuple:
            - y_test_original (np.ndarray): Actual target values from the test set.
            - predicted_courier_number_original (np.ndarray): Predicted values
            - transformed back to the original scale.
            - start_date_str (str): Start date of predictions in "YYYY-MM-DD" format.
    """
    test_target, _, test_set_scaled, scaler = load_features_target()

    x_test = []
    y_test_original = []
    test_target = test_target.reshape(-1, 1)

    for i in range(training_days, len(test_set_scaled) - n_steps + 1):
        x_test.append(test_set_scaled[i - training_days : i].flatten())
        y_test_original.append(test_target[i, 0])

    x_test = np.array(x_test)
    y_test_original = np.array(y_test_original)

    model = load(model_path)
    predicted_courier_number = model.predict(x_test)

    predicted_full = np.zeros((predicted_courier_number.shape[0], test_set_scaled.shape[1]))
    predicted_full[:, 0] = predicted_courier_number.flatten()
    predicted_original = scaler.inverse_transform(predicted_full)
    predicted_courier_number_original = predicted_original[:, 0]

    split_date_dt = datetime.strptime(split_date, "%Y-%m-%d")
    start_date = split_date_dt + timedelta(training_days)
    start_date_str = start_date.strftime("%Y-%m-%d")

    integer_predicted_courier_number = np.round(predicted_courier_number_original[training_days:])

    return (
        y_test_original[training_days:],
        integer_predicted_courier_number,
        start_date_str,
    )


app = typer.Typer()


@app.command()
def main(training_days: int = 40, n_steps: int = 1):
    """
    Main function to evaluate both LSTM and Linear Regression models.

    Parameters:
        training_days (int): Number of past days to use as input for predictions.
        n_steps (int): Number of future steps to predict.

    Returns:
        None
    """
    logger.info("Testing the LSTM model...")
    model_name = f"model_LSTM_featureDays_{training_days}_steps_{n_steps}.h5"
    model_path = MODELS_DIR / model_name

    (
        y_test_original,
        predicted_courier_number_original,
        start_date_str,
    ) = eval_lstm_model(model_path, training_days, n_steps)
    mae, _, rmse, snr, r2 = calculate_metrics(y_test_original, predicted_courier_number_original)
    logger.info(f"LSTM Metrics: MAE={mae:.2f}, R2={r2:.2f}, RMSE={rmse:.2f}, SNR={snr:.2f}")

    plot_prediction(
        y_test_original,
        predicted_courier_number_original,
        start_date_str,
        figure_title="LSTM Prediction Next Day",
        figure_path=FIGURES_DIR / "plot_LSTM_next_day.png",
    )
    logger.info("LSTM model testing complete.")

    logger.info("Testing the Linear Regression model...")
    model_name = f"model_LR_featureDays_{training_days}_steps_{n_steps}.joblib"
    model_path = MODELS_DIR / model_name

    y_test_original, predicted_courier_number_original, start_date_str = eval_lr_model(
        model_path, training_days, n_steps
    )
    mae, mse, rmse, snr, r2 = calculate_metrics(y_test_original, predicted_courier_number_original)
    logger.info(f"LR Metrics: MAE={mae:.2f}, R2={r2:.2f}, RMSE={rmse:.2f}, SNR={snr:.2f}")

    plot_prediction(
        y_test_original,
        predicted_courier_number_original,
        start_date_str,
        figure_title="Linear Regression Prediction Next Day",
        figure_path=FIGURES_DIR / "plot_LR_next_day.png",
    )
    logger.info("Linear Regression model testing complete.")


if __name__ == "__main__":
    app()
