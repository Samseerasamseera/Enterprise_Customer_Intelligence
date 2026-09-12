import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


def load_data(file_path):
    """Load feature-engineered customer data."""
    return pd.read_csv(file_path)


def prepare_data(df):
    """Separate features and target."""

    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]

    return X, y


def evaluate_model(model_name, model, X_test, y_test):
    """Evaluate a trained model."""

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(
            y_test, probabilities
        )
    }

    print("\n" + "=" * 50)
    print(f"{model_name} Results")
    print("=" * 50)

    for metric, value in metrics.items():
        if metric != "Model":
            print(f"{metric}: {value:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

    return metrics


def show_feature_importance(model, feature_names):
    """Display Random Forest feature importance."""

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    print("\n" + "=" * 50)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 50)

    print(importance_df.to_string(index=False))

    # Save feature importance
    importance_df.to_csv(
        "data/processed/feature_importance.csv",
        index=False
    )

    # Visualization
    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.title("Random Forest Feature Importance")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.show()

    return importance_df


if __name__ == "__main__":

    print("Loading dataset...")

    df = load_data(
        "data/processed/customer_features.csv"
    )

    X, y = prepare_data(df)

    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Calculate imbalance ratio for XGBoost
    scale_pos_weight = (
        (y_train == 0).sum() /
        (y_train == 1).sum()
    )

    print(
        f"\nXGBoost scale_pos_weight: "
        f"{scale_pos_weight:.2f}"
    )

    # Logistic Regression
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                class_weight="balanced",
                random_state=42,
                max_iter=1000
            )
        )
    ])

    # Random Forest
    random_forest_model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    # XGBoost
    xgboost_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss"
    )

    # Train Logistic Regression
    print("\nTraining Logistic Regression...")

    logistic_model.fit(X_train, y_train)

    logistic_metrics = evaluate_model(
        "Logistic Regression",
        logistic_model,
        X_test,
        y_test
    )

    # Train Random Forest
    print("\nTraining Random Forest...")

    random_forest_model.fit(
        X_train,
        y_train
    )

    random_forest_metrics = evaluate_model(
        "Random Forest",
        random_forest_model,
        X_test,
        y_test
    )

    # Train XGBoost
    print("\nTraining XGBoost...")

    xgboost_model.fit(
        X_train,
        y_train
    )

    xgboost_metrics = evaluate_model(
        "XGBoost",
        xgboost_model,
        X_test,
        y_test
    )

    # Model comparison
    comparison = pd.DataFrame([
        logistic_metrics,
        random_forest_metrics,
        xgboost_metrics
    ])

    print("\n" + "=" * 60)
    print("FINAL MODEL COMPARISON")
    print("=" * 60)

    print(
        comparison.to_string(
            index=False
        )
    )

    # Feature importance
    show_feature_importance(
        random_forest_model,
        X.columns
    )