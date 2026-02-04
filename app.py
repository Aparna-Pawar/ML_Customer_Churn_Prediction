from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load model and preprocessor
model = joblib.load("model/churn_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

app = FastAPI(title="Customer Churn Prediction API")

# Define input schema
class CustomerData(BaseModel):
    gender: str
    senior_citizen: int
    partner: str
    dependents: str
    phone_service: str
    multiple_lines: str
    internet_service: str
    online_security: str
    online_backup: str
    device_protection: str
    tech_support: str
    streaming_tv: str
    streaming_movies: str
    contract: str
    paperless_billing: str
    payment_method: str
    monthly_charges: float
    total_charges: float
    tenure_group: int


@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
def predict_churn(data: CustomerData):
    df = pd.DataFrame([data.dict()])
    X = preprocessor.transform(df)
    prob = model.predict_proba(X)[0][1]

    return {
        "churn_probability": round(float(prob), 3),
        "churn_prediction": int(prob > 0.5)
    }
