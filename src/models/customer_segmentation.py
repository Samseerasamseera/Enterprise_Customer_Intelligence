import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def main():

    # ---------------------------------------------------------
    # 1. Load RFM data
    # ---------------------------------------------------------

    input_path = "data/processed/customer_rfm.csv"
    output_path = "data/processed/customer_segments.csv"

    df = pd.read_csv(input_path)

    # RFM features
    features = [
        "recency",
        "frequency",
        "monetary"
    ]

    X = df[features].copy()

    # ---------------------------------------------------------
    # 2. Scale RFM features
    # ---------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    # ---------------------------------------------------------
    # 3. Find optimal number of clusters
    # ---------------------------------------------------------

    print("=" * 60)
    print("K-MEANS CUSTOMER SEGMENTATION")
    print("=" * 60)

    print("\nTesting different numbers of clusters...")

    silhouette_scores = {}

    for k in range(2, 8):

        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = kmeans.fit_predict(X_scaled)

        score = silhouette_score(
            X_scaled,
            labels
        )

        silhouette_scores[k] = score

        print(
            f"K = {k} | "
            f"Silhouette Score = {score:.4f}"
        )

    # ---------------------------------------------------------
    # 4. Select best K
    # ---------------------------------------------------------

    best_k = max(
        silhouette_scores,
        key=silhouette_scores.get
    )

    print("\n" + "=" * 60)
    print("BEST NUMBER OF CLUSTERS")
    print("=" * 60)

    print(f"Best K: {best_k}")
    print(
        f"Best Silhouette Score: "
        f"{silhouette_scores[best_k]:.4f}"
    )

    # ---------------------------------------------------------
    # 5. Train final K-Means model
    # ---------------------------------------------------------

    kmeans = KMeans(
        n_clusters=best_k,
        random_state=42,
        n_init=10
    )

    df["segment"] = kmeans.fit_predict(X_scaled)

    # ---------------------------------------------------------
    # 6. Segment profiling
    # ---------------------------------------------------------

    segment_profile = df.groupby("segment").agg(
        customers=("customer_id", "count"),
        avg_recency=("recency", "mean"),
        avg_frequency=("frequency", "mean"),
        avg_monetary=("monetary", "mean"),
        churn_rate=("churn", "mean")
    ).reset_index()

    segment_profile["churn_rate"] *= 100

    print("\n" + "=" * 60)
    print("CUSTOMER SEGMENT PROFILE")
    print("=" * 60)

    print(
        segment_profile.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------
    # 7. Save segmented customers
    # ---------------------------------------------------------

    df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 60)
    print("SEGMENTATION COMPLETED")
    print("=" * 60)

    print(f"Customers: {len(df)}")
    print(f"Segments: {best_k}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()