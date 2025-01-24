from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from wolt_test_assignment.modeling.train import train_linear_regression, train_rnn

# from wolt_test_assignment.config import PROCESSED_DATA_DIR, MODELS_DIR

# Mock the PROCESSED_DATA_DIR and MODELS_DIR
PROCESSED_DATA_DIR = "path/to/processed_data"
MODELS_DIR = "path/to/models"


@patch("wolt_test_assignment.modeling.train.load_scaled_data")
@patch("wolt_test_assignment.modeling.train.LinearRegression")
@patch("wolt_test_assignment.modeling.train.MultiOutputRegressor")
def test_train_linear_regression(
    mock_multi_output_regressor, mock_linear_regression, mock_load_scaled_data
):
    # Arrange
    mock_load_scaled_data.return_value = (np.random.rand(100, 10), None, None)
    mock_linear_regression.return_value = MagicMock()
    mock_multi_output_regressor.return_value = MagicMock()

    training_days = 40
    n_steps = 1

    # Act
    model = train_linear_regression(training_days, n_steps)

    # Assert
    assert model is not None
    if n_steps == 1:
        mock_linear_regression.assert_called_once()
    else:
        mock_multi_output_regressor.assert_called_once()


@patch("wolt_test_assignment.modeling.train.load_scaled_data")
@patch("wolt_test_assignment.modeling.train.create_regressor")
def test_train_rnn(mock_create_regressor, mock_load_scaled_data):
    # Arrange
    mock_load_scaled_data.return_value = (
        np.random.rand(100, 10),
        np.random.rand(20, 10),
        MagicMock(),
    )
    mock_create_regressor.return_value = MagicMock()

    epochs = 10
    batch_size = 32
    training_days = 40
    n_steps = 1

    # Act
    model = train_rnn(epochs, batch_size, training_days, n_steps)

    # Assert
    assert model is not None
    mock_create_regressor.assert_called_once()


if __name__ == "__main__":
    pytest.main()
