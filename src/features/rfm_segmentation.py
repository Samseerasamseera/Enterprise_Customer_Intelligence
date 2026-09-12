import pandas as pd


def create_rfm_features(df):
    """
    Create RFM features for customer segmentation.

    R = Recency
    F = Frequency
    M = Monetary Value
    """

    df = df.copy()

    # ---------------------------------------------------------
    # Recency
    # ---------------------------------------------------------
    # In this synthetic dataset we don't have transaction dates.
    # Therefore, use tenure as a proxy for customer activity.
    #
    # Higher tenure -> lower recency score (more established)
    # ---------------------------------------------------------

    max_tenure = df["tenure_months"].max()

    df["recency"] = max_tenure - df["tenure_months"]

    # ---------------------------------------------------------
    # Frequency
    # ---------------------------------------------------------

    df["frequency"] = df["total_purchases"]

    # ---------------------------------------------------------
    # Monetary Value
    # ---------------------------------------------------------

    df["monetary"] = (
        df["total_purchases"]
        * df["avg_transaction_value"]
    )

    return df


def main():

    input_path = "data/processed/customer_features.csv"
    output_path = "data/processed/customer_rfm.csv"

    # Load data
    df = pd.read_csv(input_path)

    # Create RFM features
    df = create_rfm_features(df)

    # Select useful columns
    rfm_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "churn"
    ]

    rfm_df = df[rfm_columns]

    # Save RFM dataset
    rfm_df.to_csv(output_path, index=False)

    print("=" * 60)
    print("RFM CUSTOMER SEGMENTATION")
    print("=" * 60)

    print(f"Customers: {len(rfm_df)}")
    print(f"RFM features created: Recency, Frequency, Monetary")
    print(f"Saved to: {output_path}")

    print("\nSample RFM data:")
    print(rfm_df.head())

    print("=" * 60)


if __name__ == "__main__":
    main()