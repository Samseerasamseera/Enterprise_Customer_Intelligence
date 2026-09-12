import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# Load data
df = pd.read_csv(
    "data/processed/customer_features.csv"
)

# Remove ID and separate target
X = df.drop(columns=["churn", "customer_id"])
y = df["churn"]


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Build Logistic Regression pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            random_state=42,
            max_iter=1000
        )
    )
])


# Train
model.fit(X_train, y_train)


# Get probability of churn
y_probability = model.predict_proba(X_test)[:, 1]


print("=" * 60)
print("THRESHOLD TUNING")
print("=" * 60)


results = []

# Test different thresholds
thresholds = np.arange(0.20, 0.71, 0.05)

for threshold in thresholds:

    y_pred = (
        y_probability >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Threshold": round(threshold, 2),
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


results_df = pd.DataFrame(results)

print(
    results_df.to_string(index=False)
)


# Find threshold with highest F1
best_row = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 60)
print("BEST THRESHOLD")
print("=" * 60)

print(
    f"Threshold: {best_row['Threshold']:.2f}"
)

print(
    f"Precision: {best_row['Precision']:.4f}"
)

print(
    f"Recall: {best_row['Recall']:.4f}"
)

print(
    f"F1 Score: {best_row['F1']:.4f}"
)


# ROC-AUC does not depend on a single threshold
roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)