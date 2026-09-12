import joblib
import pandas as pd

from src.features.feature_engineering import create_features


# Load saved model
model = joblib.load("models/churn_model.pkl")

# Tuned threshold
THRESHOLD = 0.45


def predict_churn(customer_data):
    """
    Predict customer churn probability and risk level.
    """

    # Convert raw input to DataFrame
    df = pd.DataFrame([customer_data])

    # Create engineered features
    df = create_features(df)

    # Remove columns not used by the model
    df = df.drop(columns=["customer_id", "churn"], errors="ignore")

    # Predict probability
    probability = model.predict_proba(df)[0][1]

    # Apply tuned threshold
    prediction = int(probability >= THRESHOLD)

    # Risk classification
    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.45:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return probability, prediction, risk


def main():

    print("=" * 60)
    print("ENTERPRISE CUSTOMER CHURN PREDICTION")
    print("=" * 60)

    # Get customer information
    age = float(input("Age: "))
    tenure_months = float(input("Tenure (months): "))
    monthly_charges = float(input("Monthly charges: "))
    total_purchases = float(input("Total purchases: "))
    avg_transaction_value = float(input("Average transaction value: "))
    support_tickets = float(input("Support tickets: "))

    # Create raw customer record
    customer = {
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_purchases": total_purchases,
        "avg_transaction_value": avg_transaction_value,
        "support_tickets": support_tickets
    }

    # Predict
    probability, prediction, risk = predict_churn(customer)

    print()
    print("=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)

    print(f"Churn Probability : {probability:.2%}")

    if prediction == 1:
        print("Prediction         : WILL CHURN")
    else:
        print("Prediction         : WILL NOT CHURN")

    print(f"Risk Level         : {risk}")

    print("=" * 60)


if __name__ == "__main__":
    main()