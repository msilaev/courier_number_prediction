import numpy as np
import joblib
from wolt_test_assignment.config import MODELS_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, INTERIM_DATA_DIR, SPLIT_DATE
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_features_target():
    """
    Load the scaled features scaler from the processed data directory.

    Returns:
    tuple: Scaled total, training, and test sets, and the scaler.
    """
    total_set_scaled = np.load(PROCESSED_DATA_DIR / "total_feature_set_scaled.npy")
    training_set_scaled = np.load(PROCESSED_DATA_DIR / "training_feature_set_scaled.npy")
    test_set_scaled = np.load(PROCESSED_DATA_DIR / "test_feature_set_scaled.npy")
    test_target = np.load(PROCESSED_DATA_DIR / "test_target_set.npy")

    scaler = joblib.load(PROCESSED_DATA_DIR / "scaler.pkl")

    return test_target, training_set_scaled, test_set_scaled, scaler


def calculate_snr(y_true, y_pred):
    """
    Calculate the Signal-to-Noise Ratio (SNR) in decibels.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    float: SNR value in decibels.
    """
    signal_variance = np.var(y_true)
    noise_variance = np.var(y_true - y_pred)
    #snr = 10 * np.log10(signal_variance / noise_variance)
    
    snr = 10 * np.log10( np.mean(y_true**2) / np.mean( (y_true- y_pred)**2) )
    
    return snr

def calculate_metrics(y_true, y_pred):
    """
    Calculate evaluation metrics.

    Parameters:
    y_true (array-like): True values.
    y_pred (array-like): Predicted values.

    Returns:
    dict: Dictionary containing MAE, MSE, and RMSE.
    """
    
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, rmse, calculate_snr(y_true, y_pred), r2

#   return f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}, SNR: {calculate_snr(y_true, y_pred)}"