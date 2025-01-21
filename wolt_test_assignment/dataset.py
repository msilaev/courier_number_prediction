from pathlib import Path
import typer
import pandas as pd
from loguru import logger

from wolt_test_assignment.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, INTERIM_DATA_DIR


def load_data(file_path):
    """
    Load the dataset from a CSV file and parse dates.

    Parameters:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pd.DataFrame: The loaded dataset with the 'date' column converted to datetime.
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def clean_data(df, anomaly_threshold=500):
    """
    Preprocess the dataset by removing anomalies and handling missing values.

    Parameters:
        df (pd.DataFrame): The dataset to preprocess.
        anomaly_threshold (int, optional): Threshold value to identify and remove anomalies 
                                           in the 'courier_partners_online' column. 
                                           Defaults to 500.

    Returns:
        pd.DataFrame: A cleaned dataset with:
                      - Anomalies removed based on the threshold.
                      - Missing values forward-filled.
                      - Day-of-week feature encoded as one-hot vectors.
    """
    df_cleaned = df.copy(deep=True)
    df_cleaned.loc[df['courier_partners_online'] > anomaly_threshold, 'courier_partners_online'] = None
    df_cleaned = df_cleaned.ffill()
    df_cleaned['day_of_week'] = df_cleaned['date'].dt.dayofweek
    df_cleaned = pd.get_dummies(df_cleaned, columns=['day_of_week'])
    return df_cleaned


app = typer.Typer()

@app.command()
def main(
    input_path: Path = RAW_DATA_DIR / "daily_cp_activity_dataset.csv",
    output_path: Path = INTERIM_DATA_DIR / "daily_cp_activity_dataset.csv"
):
    """
    Main function to process and clean the dataset.

    Parameters:
        input_path (Path): Path to the input CSV file containing raw data. 
                           Defaults to 'daily_cp_activity_dataset.csv' in RAW_DATA_DIR.
        output_path (Path): Path to save the processed dataset. 
                            Defaults to 'daily_cp_activity_dataset.csv' in INTERIM_DATA_DIR.

    Steps:
        1. Load the dataset from the input CSV file.
        2. Clean the data by removing anomalies, filling missing values, and adding one-hot encoding.
        3. Save the cleaned dataset to the specified output path.
    """
    logger.info("Processing dataset...")

    df = load_data(input_path)
    df_cleaned = clean_data(df)
    df_cleaned.to_csv(output_path, index=False)

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
