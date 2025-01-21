import pytest
from wolt_test_assignment.dataset import load_data  # Replace with actual function names
from wolt_test_assignment.config import INTERIM_DATA_DIR


def test_load_data():
    # Arrange
    file_path = INTERIM_DATA_DIR / "daily_cp_activity_dataset.csv"
    expected_columns = [
        "date",
        "courier_partners_online",
        "temperature",
        "relative_humidity",
        "precipitation",
        "day_of_week_0",
        "day_of_week_1",
        "day_of_week_2",
        "day_of_week_3",
        "day_of_week_4",
        "day_of_week_5",
        "day_of_week_6",
    ]

    # Act
    data = load_data(file_path)

    # Assert
    assert list(data.columns) == expected_columns
    assert not data.empty
