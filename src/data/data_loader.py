import pandas as pd
from pathlib import Path


class DataLoader:
    """Load and perform basic validation on customer data."""

    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load_data(self):
        """Load the CSV dataset."""

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.file_path}"
            )

        df = pd.read_csv(self.file_path)

        print("Dataset loaded successfully!")
        print(f"Shape: {df.shape}")

        return df

    def validate_data(self, df):
        """Perform basic data validation."""

        print("\n--- Data Validation Report ---")

        # Check duplicates
        duplicate_count = df.duplicated().sum()
        print(f"Duplicate rows: {duplicate_count}")

        # Check missing values
        print("\nMissing Values:")
        print(df.isnull().sum())

        # Check data types
        print("\nData Types:")
        print(df.dtypes)

        # Basic statistics
        print("\nStatistical Summary:")
        print(df.describe())

        return {
            "duplicates": duplicate_count,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.astype(str).to_dict()
        }


if __name__ == "__main__":

    loader = DataLoader(
        "data/raw/customer_data.csv"
    )

    df = loader.load_data()

    validation_report = loader.validate_data(df)

    print("\n--- Validation Completed ---")