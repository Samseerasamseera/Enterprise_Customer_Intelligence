import pandas as pd


def create_features(df):
    """
    Create engineered features for customer data.
    """

    df = df.copy()

    df["estimated_lifetime_value"] = (
        df["monthly_charges"] * df["tenure_months"]
    )

    df["estimated_total_spending"] = (
        df["total_purchases"] * df["avg_transaction_value"]
    )

    df["support_ticket_rate"] = (
        df["support_tickets"] / (df["tenure_months"] + 1)
    )

    df["purchase_frequency"] = (
        df["total_purchases"] / (df["tenure_months"] + 1)
    )

    return df


if __name__ == "__main__":

    print("Starting feature engineering...")

    input_path = "data/processed/customer_data_processed.csv"
    output_path = "data/processed/customer_features.csv"

    df = pd.read_csv(input_path)

    df = create_features(df)

    df.to_csv(output_path, index=False)

    print(f"Total features: {df.shape[1]}")
    print(f"Feature data saved to: {output_path}")
    print("\nFeature engineering completed successfully!")