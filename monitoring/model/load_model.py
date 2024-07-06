# Load a model from MLflow and use it for monitoring

import os
import mlflow

tracking_uri = os.getenv('FRAUD_MODELLING_MLFLOW_TRACKING_URI')
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment('Insurance Fraud Detection')

run_id = os.getenv('FRAUD_MODELLING_MLFLOW_RUN_ID')
model_uri = mlflow.get_run(run_id).info.artifact_uri + '/balanced_rf_model'
mlflow.artifacts.download_artifacts(model_uri, dst_path='./model')
