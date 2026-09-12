import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load data
df = pd.read_csv("data/processed/customer_features.csv")

# Separate features and target
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

# Create model pipeline
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    ))
])

# Train model
model.fit(X_train, y_train)

# Create models directory
os.makedirs("models", exist_ok=True)

# Save model
model_path = "models/churn_model.pkl"
joblib.dump(model, model_path)

print("=" * 60)
print("MODEL SAVING")
print("=" * 60)
print(f"Model saved successfully!")
print(f"Location: {model_path}")
print(f"Training samples: {len(X_train)}")
print(f"Features: {X.shape[1]}")
print("=" * 60)