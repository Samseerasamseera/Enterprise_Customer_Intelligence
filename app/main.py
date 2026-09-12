from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

from src.features.feature_engineering import create_features


# Create FastAPI application
app = FastAPI(
    title="Enterprise Customer Intelligence API",
    description="Customer churn prediction API",
    version="1.0.0"
)


# Load trained model
model = joblib.load("models/churn_model.pkl")

# Tuned classification threshold
THRESHOLD = 0.45


# Request schema
class CustomerData(BaseModel):
    age: float
    tenure_months: float
    monthly_charges: float
    total_purchases: float
    avg_transaction_value: float
    support_tickets: float


@app.get("/")
def home():
    return {
        "message": "Enterprise Customer Intelligence API is running"
    }


@app.post("/predict")
def predict_churn(customer: CustomerData):

    # Convert request to DataFrame
    df = pd.DataFrame([customer.model_dump()])

    # Apply feature engineering
    df = create_features(df)

    # Predict churn probability
    probability = model.predict_proba(df)[0][1]

    # Apply tuned threshold
    prediction = int(probability >= THRESHOLD)

    # Determine risk
    if probability >= 0.70:
        risk = "HIGH"
    elif probability >= 0.45:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": "WILL CHURN" if prediction == 1 else "WILL NOT CHURN",
        "risk_level": risk,
        "threshold": THRESHOLD
    }