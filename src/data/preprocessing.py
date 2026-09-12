import pandas as pd
from pathlib import Path


class DataPreprocessor:
    """Clean and preprocess customer data."""

    def __init__(self, file_path):
        self.file_path = Path(file_path)

    def load_data(self):
        """Load the raw dataset."""

        if not self.file_path.exists():
            raise FileNotFoundError(
                f"File not found: {self.file_path}"
            )

        return pd.read_csv(self.file_path)

    def preprocess(self, df):
        """Clean the dataset."""

        print("Starting data preprocessing...")

        # Remove duplicate rows
        df = df.drop_duplicates().copy()

        # Fill missing numerical values with the median
        df["avg_transaction_value"] = (
            df["avg_transaction_value"]
            .fillna(df["avg_transaction_value"].median())
        )

        # Validate that no missing values remain
        missing_values = df.isnull().sum().sum()

        print(f"Remaining missing values: {missing_values}")
        print(f"Processed dataset shape: {df.shape}")

        return df

    def save_data(self, df, output_path):
        """Save processed data."""

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)

        print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":

    input_path = "data/raw/customer_data.csv"
    output_path = "data/processed/customer_data_processed.csv"

    preprocessor = DataPreprocessor(input_path)

    # Load data
    df = preprocessor.load_data()

    # Preprocess data
    processed_df = preprocessor.preprocess(df)

    # Save processed data
    preprocessor.save_data(processed_df, output_path)

    print("\nData preprocessing completed successfully!")