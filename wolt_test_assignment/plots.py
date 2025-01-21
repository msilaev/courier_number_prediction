from pathlib import Path

import typer
from loguru import logger
import pandas as pd

# from tqdm import tqdm

from wolt_test_assignment.config import FIGURES_DIR, PROCESSED_DATA_DIR

import matplotlib.pyplot as plt


def plot_prediction(
    y_test_original,
    predicted_courier_number_original,
    start_date,
    figure_title,
    figure_path,
):
    # def plot_prediction(y_test_original, predicted_courier_number_original):
    """
    Plot the real and predicted values for courier partners online.

    Parameters:
    y_test_original (array-like): The original test values (real values).
    predicted_courier_number_original (array-like): The predicted values from the model.

    Returns:
    None
    """

    date_range = pd.date_range(start=start_date, periods=len(y_test_original))
    plt.plot(date_range, y_test_original, color="red", marker="+", label="Real")
    plt.plot(
        date_range,
        predicted_courier_number_original,
        color="blue",
        marker="o",
        label="Predicted",
    )
    plt.title("Prediction")
    plt.xlabel("Date")
    plt.ylabel("Number of Courier Partners Online")
    plt.title(figure_title)
    plt.legend()
    plt.grid()
    plt.xticks(rotation=45)
    plt.tight_layout()
    # plt.show()
    plt.savefig(figure_path)
    plt.close()


app = typer.Typer()


@app.command()
def main(
    input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
    output_path: Path = FIGURES_DIR / "plot.png",
):
    logger.info("Generating plot from data...")

    logger.success("Plot generation complete.")


if __name__ == "__main__":
    app()
