import pickle
from fastapi import FastAPI

# loading from local so I do not need to keep the mlflow server from GCP running
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Insurance Fraud Detection API"}

@app.post("/predict")
def predict_fraud(data: dict):
    # prediction = model.predict(data)
    return {"prediction": True}