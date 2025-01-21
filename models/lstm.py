from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense


def create_regressor(input_shape, n_steps):
    """
    Create and compile an LSTM model with Dropout regularization.

    Parameters:
    input_shape (tuple): Shape of the input data (timesteps, features).

    Returns:
    model (Sequential): Compiled LSTM model.
    """
    regressor = Sequential()

    # First LSTM layer with Dropout regularisation
    regressor.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    regressor.add(Dropout(0.3))

    # Second LSTM layer with Dropout regularisation
    regressor.add(LSTM(units=80, return_sequences=True))
    regressor.add(Dropout(0.1))

    # Third LSTM layer with Dropout regularisation
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(0.2))

    # Fourth LSTM layer with Dropout regularisation
    regressor.add(LSTM(units=30))
    regressor.add(Dropout(0.3))

    # Output layer
    regressor.add(Dense(units=n_steps))

    # Compile the model
    regressor.compile(optimizer="adam", loss="mean_squared_error")

    return regressor
