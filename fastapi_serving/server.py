import pickle
import pandas as pd
from fastapi import FastAPI

# how to deploy: https://youtu.be/xZ013IgK7Ts

# loading from local so no not need to keep the mlflow server from GCP running
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def read_root():
    """Welcome message."""
    return {"message": "Welcome to the Insurance Fraud Detection API"}

@app.post("/predict")
def predict_fraud(data: pd.DataFrame):
    """Predict fraud for given data"""
    y_pred = model.predict(data)
    data_with_predictions = data.copy()
    data_with_predictions['FraudFound_P'] = y_pred
    return {"prediction": data_with_predictions}
