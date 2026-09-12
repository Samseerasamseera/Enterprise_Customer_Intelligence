import pandas as pd
import numpy as np
from pathlib import Path


def generate_customer_data(n_customers=5000, random_state=42):
    """Generate a realistic synthetic customer dataset."""

    np.random.seed(random_state)

    customer_ids = range(100001, 100001 + n_customers)

    ages = np.random.randint(18, 75, n_customers)
    tenure_months = np.random.randint(1, 121, n_customers)

    monthly_charges = np.round(
        np.random.uniform(500, 5000, n_customers), 2
    )

    total_purchases = np.random.randint(1, 100, n_customers)

    avg_transaction_value = np.round(
        np.random.uniform(200, 10000, n_customers), 2
    )

    support_tickets = np.random.poisson(2, n_customers)

    # Create churn probability
    churn_probability = (
        0.15
        + (support_tickets * 0.03)
        - (tenure_months * 0.001)
    )

    churn_probability = np.clip(
        churn_probability, 0.05, 0.85
    )

    churn = np.random.binomial(1, churn_probability)

    data = pd.DataFrame({
        "customer_id": customer_ids,
        "age": ages,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_purchases": total_purchases,
        "avg_transaction_value": avg_transaction_value,
        "support_tickets": support_tickets,
        "churn": churn
    })

    # Introduce some missing values for preprocessing
    missing_indices = np.random.choice(
        data.index,
        size=int(n_customers * 0.05),
        replace=False
    )

    data.loc[missing_indices, "avg_transaction_value"] = np.nan

    return data


if __name__ == "__main__":

    data = generate_customer_data()

    # Create output directory if it doesn't exist
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "customer_data.csv"

    data.to_csv(output_path, index=False)

    print(f"Dataset created successfully!")
    print(f"Shape: {data.shape}")
    print(f"Saved to: {output_path}")

    print("\nFirst 5 rows:")
    print(data.head())