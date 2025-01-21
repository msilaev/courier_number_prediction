import pytest
from wolt_test_assignment.dataset import load_data  # Replace with actual function names
from wolt_test_assignment.config import RAW_DATA_DIR


def test_load_data():
    # Arrange
    file_path = RAW_DATA_DIR / "daily_cp_activity_dataset.csv"
    expected_columns = [
        "date",
        "courier_partners_online",
        "temperature",
        "relative_humidity",
        "precipitation",
    ]

    # Act
    data = load_data(file_path)

    # Assert
    assert list(data.columns) == expected_columns
    assert not data.empty
