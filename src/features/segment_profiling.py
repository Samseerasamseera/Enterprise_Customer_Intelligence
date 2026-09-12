import pandas as pd


def profile_segments():

    input_path = "data/processed/customer_segments.csv"
    output_path = "data/processed/customer_segments_profiled.csv"

    # Load segmented customers
    df = pd.read_csv(input_path)

    # ---------------------------------------------------------
    # 1. Create segment profile
    # ---------------------------------------------------------

    profile = (
        df.groupby("segment")
        .agg(
            customers=("customer_id", "count"),
            avg_recency=("recency", "mean"),
            avg_frequency=("frequency", "mean"),
            avg_monetary=("monetary", "mean"),
            churn_rate=("churn", "mean")
        )
        .reset_index()
    )

    # Convert churn rate to percentage
    profile["churn_rate"] = profile["churn_rate"] * 100

    # ---------------------------------------------------------
    # 2. Identify high-value segment
    # ---------------------------------------------------------

    high_value_segment = profile.loc[
        profile["avg_monetary"].idxmax(),
        "segment"
    ]

    # ---------------------------------------------------------
    # 3. Assign business-friendly labels
    # ---------------------------------------------------------

    segment_names = {}

    for _, row in profile.iterrows():

        segment = row["segment"]

        if segment == high_value_segment:
            segment_names[segment] = "High Value Customers"
        else:
            segment_names[segment] = "Standard Value Customers"

    # Apply labels
    df["segment_name"] = df["segment"].map(segment_names)

    # ---------------------------------------------------------
    # 4. Save profiled dataset
    # ---------------------------------------------------------

    df.to_csv(
        output_path,
        index=False
    )

    # ---------------------------------------------------------
    # 5. Display results
    # ---------------------------------------------------------

    print("=" * 60)
    print("CUSTOMER SEGMENT PROFILING")
    print("=" * 60)

    print("\nSEGMENT PROFILE")

    for _, row in profile.iterrows():

        segment = row["segment"]

        print(
            f"\nSegment: {segment}"
        )

        print(
            f"Customers: {int(row['customers'])}"
        )

        print(
            f"Average Recency: "
            f"{row['avg_recency']:.2f}"
        )

        print(
            f"Average Frequency: "
            f"{row['avg_frequency']:.2f}"
        )

        print(
            f"Average Monetary Value: "
            f"{row['avg_monetary']:.2f}"
        )

        print(
            f"Churn Rate: "
            f"{row['churn_rate']:.2f}%"
        )

        print(
            f"Business Label: "
            f"{segment_names[segment]}"
        )

    print("\n" + "=" * 60)
    print("SEGMENT PROFILING COMPLETED")
    print("=" * 60)

    print(
        f"Saved to: {output_path}"
    )


if __name__ == "__main__":
    profile_segments()