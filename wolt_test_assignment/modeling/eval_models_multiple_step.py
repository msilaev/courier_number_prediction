# Standard library imports
import os
import warnings
from datetime import datetime, timedelta

# Related third-party imports
import absl.logging
import numpy as np
import typer
from joblib import load
from keras.models import load_model
from loguru import logger

# Local application/library-specific imports
from wolt_test_assignment.config import FIGURES_DIR, MODELS_DIR, SPLIT_DATE
from wolt_test_assignment.modeling.utils import (
    calculate_metrics,
    load_features_target,
    load_and_prepare_data,
    load_and_prepare_data_lstm,
)

from wolt_test_assignment.plots import plot_prediction

# Suppress TensorFlow INFO and WARNING logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Suppress `absl` logging
absl.logging.set_verbosity(absl.logging.ERROR)

# Suppress specific Python warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=r"Compiled the loaded model.*")


def eval_lr_model(
    model_name: str = "model_LR_multioutput.joblib",
    training_days: int = 40,
    n_steps: int = 20,
    start_ind: int = 0,
):
    """
    Evaluate the Linear Regression model using MultiOutputRegressor.

    Parameters:
    - model_name (str): Name of the saved model file.
    - training_days (int): Number of days used as input for predictions.
    - n_steps (int): Number of steps to predict into the future.
    - start_ind (int): Index to start evaluation from.

    Returns:
    - tuple:
        - y_test_original (ndarray): Ground truth values.
        - predicted_courier_number_original (ndarray): Predicted values.
        - start_date_str (str): Evaluation start date in 'YYYY-MM-DD' format.
    """

    # Load and prepare data
    x_test, test_target, scaler, feature_number = load_and_prepare_data(training_days, n_steps)

    y_test_original = test_target[
        start_ind + training_days : start_ind + training_days + n_steps
    ].flatten()

    # Load the trained model
    model_path = MODELS_DIR / model_name
    model = load(model_path)  # Load MultiOutputRegressor(LinearRegression())

    # Make predictions on the test set
    predicted_courier_number = model.predict(x_test)

    predicted_courier_number = predicted_courier_number[start_ind, :].flatten()

    # Create a placeholder array for inverse transform
    predicted_full = np.zeros((predicted_courier_number.shape[0], feature_number))

    predicted_full[:, 0] = predicted_courier_number.flatten()
    #################################

    predicted_original = scaler.inverse_transform(predicted_full)

    # Extract the original scale predictions for 'courier_partners_online'
    predicted_courier_number_original = predicted_original[:, 0]

    # integer_predicted_courier_number = np.round(predicted_courier_number_original)

    # Calculate the start date by adding the timedelta
    start_date = datetime.strptime(SPLIT_DATE, "%Y-%m-%d") + timedelta(
        days=training_days + start_ind
    )
    start_date_str = start_date.strftime("%Y-%m-%d")

    return y_test_original, predicted_courier_number_original, start_date_str


def eval_lstm_model(
    model_name: str = "model_RNN.h5",
    training_days: int = 40,
    n_steps: int = 20,
    start_ind: int = 0,
):
    """
    Evaluate the LSTM model on the test dataset.

    Parameters:
    - model_name (str): Name of the saved LSTM model file.
    - training_days (int): Number of days used as input for predictions.
    - n_steps (int): Number of steps to predict into the future.
    - start_ind (int): Index to start evaluation from.

    Returns:
    - tuple:
        - y_test_original (ndarray): Ground truth values.
        - predicted_courier_number_original (ndarray): Predicted values.
        - start_date_str (str): Evaluation start date in 'YYYY-MM-DD' format.
    """

    # Load and prepare test features and target
    x_test, test_target, scaler, feature_number = load_and_prepare_data_lstm(
        training_days, n_steps
    )

    y_test_original = test_target[training_days : len(test_target) - n_steps + 1].flatten()

    y_test_original = np.array(y_test_original)

    # Load the trained model
    model_path = MODELS_DIR / model_name
    model = load_model(model_path)

    predicted_courier_number = model.predict(x_test, verbose=0)

    predicted_courier_number = predicted_courier_number[start_ind, :].flatten()

    y_test_original = test_target[
        start_ind + training_days : start_ind + training_days + n_steps
    ].flatten()

    # Create a placeholder array for inverse transform
    predicted_full = np.zeros((predicted_courier_number.shape[0], feature_number))
    predicted_full[:, 0] = predicted_courier_number.flatten()

    # Inverse transform the predictions
    predicted_original = scaler.inverse_transform(predicted_full)

    # Extract the original scale predictions for 'courier_partners_online'
    predicted_courier_number_original = predicted_original[:, 0]

    # Convert SPLIT_DATE to a datetime object
    start_date = datetime.strptime(SPLIT_DATE, "%Y-%m-%d") + timedelta(
        days=training_days + start_ind
    )
    #   Convert start_date back to a string if needed
    start_date_str = start_date.strftime("%Y-%m-%d")

    return y_test_original, predicted_courier_number_original, start_date_str


app = typer.Typer()


@app.command()
def main(training_days: int = 40, n_steps: int = 20, start_ind: int = 0):
    """
    Evaluate and compare LSTM and Linear Regression models.

    Parameters:
    - training_days (int): Number of days used as input for predictions.
    - n_steps (int): Number of steps to predict into the future.
    - start_ind (int): Index to start evaluation from.

    Runs evaluations for both the LSTM and Linear Regression models,
    calculates metrics, and generates prediction plots.
    """

    test_target, _, _, _ = load_features_target()

    number_intervals = len(test_target) - training_days - n_steps + 1

    logger.info("testing the LSTM model...")

    #    model_name = f"model_LSTM_step_{n_steps}.h5"
    model_name = f"model_LSTM_featureDays_{training_days}_steps_{n_steps}.h5"
    model_path = MODELS_DIR / model_name

    mae_list = []
    mse_list = []
    rmse_list = []
    snr_list = []
    r2_list = []

    for start_ind_ in range(0, number_intervals):

        y_test_original, predicted_courier_number_original, start_date = eval_lstm_model(
            model_path, training_days, n_steps, start_ind_
        )

        mae, mse, rmse, snr, r2 = calculate_metrics(
            y_test_original, predicted_courier_number_original
        )
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        snr_list.append(snr)
        r2_list.append(r2)

        if start_ind_ % 10 == 0:
            plot_prediction(
                y_test_original,
                predicted_courier_number_original,
                start_date,
                figure_title=f"LSTM prediction {n_steps} days forward",
                figure_path=(
                    FIGURES_DIR
                    / f"plot_LSTM_featureDays_{training_days}_steps_{n_steps}_days_interval_{start_ind_}.png"
                ),
            )

    mae = np.array(mae_list)
    mse = np.array(mse_list)
    rmse = np.array(rmse_list)
    snr = np.array(snr_list)
    r2 = np.array(r2_list)

    # metrics = f"MAE: {np.mean(mae):.2f}, MSE: {np.mean(mse):.2f}, R2: {np.mean(r2):.2f}, SNR: {np.mean(snr):.2f}"

    metrics = (
        f"MAE: {np.mean(mae):.2f}, "
        f"MSE: {np.mean(mse):.2f}, "
        f"RMSE: {np.mean(rmse):.2f}, "
        f"R2: {np.mean(r2):.2f}, "
        f"SNR: {np.mean(snr):.2f}"
    )

    logger.info("LSTM metrics " + metrics)
    logger.info("testing the LSTM model done")

    #########################################

    logger.info("testing the Linear regression model...")

    model_name = f"model_LR_featureDays_{training_days}_steps_{n_steps}.joblib"
    model_path = MODELS_DIR / model_name

    mae_list = []
    mse_list = []
    rmse_list = []
    snr_list = []
    r2_list = []

    for start_ind in range(0, number_intervals):

        y_test_original, predicted_courier_number_original, start_date_str = eval_lr_model(
            model_path, training_days, n_steps, start_ind
        )

        mae, mse, rmse, snr, r2 = calculate_metrics(
            y_test_original, predicted_courier_number_original
        )
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
        snr_list.append(snr)
        r2_list.append(r2)

        if start_ind % 10 == 0:
            plot_prediction(
                y_test_original,
                predicted_courier_number_original,
                start_date_str,
                figure_title=f"Linear regression prediction {n_steps} days forward",
                figure_path=FIGURES_DIR / f"plot_LR_featureDays_{training_days}_"
                f"steps_{n_steps}_days_interval_{start_ind}.png",
            )

    mae = np.array(mae_list)
    mse = np.array(mse_list)
    rmse = np.array(rmse_list)
    snr = np.array(snr_list)
    r2 = np.array(r2_list)

    metrics = (
        f"MAE: {np.mean(mae):.2f}, "
        f"MSE: {np.mean(mse):.2f}, "
        f"RMSE: {np.mean(rmse):.2f}, "
        f"R2: {np.mean(r2):.2f}, "
        f"SNR: {np.mean(snr):.2f}"
    )

    logger.info("LR metrics " + metrics)

    logger.info("testing the Linear regression model done")


if __name__ == "__main__":
    app()
