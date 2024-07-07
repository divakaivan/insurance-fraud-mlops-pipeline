import os
import mlflow
from fastapi import FastAPI

tracking_uri = os.getenv('FRAUD_MODELLING_MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment('Insurance Fraud Detection')

run_id = os.getenv('FRAUD_MODELLING_MLFLOW_RUN_ID')
logged_model = f'runs:/{run_id}/balanced_rf_model'

model = mlflow.pyfunc.load_model(logged_model)
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Insurance Fraud Detection API"}

@app.post("/predict")
def predict_fraud(data: dict):
    # prediction = model.predict(data)
    return {"prediction": True}