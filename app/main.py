from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd


# Create FastAPI app
app = FastAPI(
    title="Banking Fraud Detection API",
    description="API for detecting fraudulent banking transactions",
    version="1.0.0"
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TransactionRequest(BaseModel):
    Time: float = Field(..., ge=0)
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., ge=0)


# Load trained model
model = joblib.load(
    PROJECT_ROOT / "models" / "fraud_detection_xgboost.joblib"
)

# Load feature names
feature_names = joblib.load(
    PROJECT_ROOT / "models" / "feature_names.joblib"
)


@app.get("/")
def home():
    return {
        "message": "Banking Fraud Detection API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "XGBoost"
    }


@app.post("/predict")
def predict(transaction: TransactionRequest):
    data = transaction.model_dump()
    data["Time_Hours"] = data["Time"] / 3600

    features = pd.DataFrame([data])[feature_names]
    fraud_probability = float(model.predict_proba(features)[0][1])

    return {
        "prediction": "FRAUD" if fraud_probability >= 0.5 else "GENUINE",
        "fraud_probability": round(fraud_probability, 6),
        "threshold": 0.5,
    }