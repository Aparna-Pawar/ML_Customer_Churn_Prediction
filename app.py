
##customer churn frontend
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
import pickle
from mlflow.tracking import MlflowClient

# ===============================
# CONFIG
# ===============================
RUN_ID = "df0b39914755441da96e5f5bf48c268c"
MODEL_URI = f"runs:/{RUN_ID}/model"

app = FastAPI(title="Customer Churn Prediction API")

# ===============================
# LOAD MODEL ON STARTUP
# ===============================
@app.on_event("startup")
def load_model():
    global model, preprocessor

    # Load model from MLflow
    model = mlflow.sklearn.load_model(MODEL_URI)

    # Download preprocessor artifact
    client = MlflowClient()
    client.download_artifacts(RUN_ID, "preprocessor.pkl", ".")

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    print("✅ Model and preprocessor loaded successfully")


# ===============================
# INPUT SCHEMA
# ===============================
class ChurnInput(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str
    InternetService: str


# ===============================
# HEALTH CHECK
# ===============================
@app.get("/")
def health_check():
    return {"status": "API is running"}


# ===============================
# PREDICTION ENDPOINT
# ===============================
@app.post("/predict")
def predict_churn(data: ChurnInput):

    input_df = pd.DataFrame([data.dict()])

    # Preprocess
    X_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(X_processed)[0]
    probability = model.predict_proba(X_processed)[0][1]

    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(probability, 3),
        "note": "Recall-optimized model (focus on minimizing false negatives)"
    }
