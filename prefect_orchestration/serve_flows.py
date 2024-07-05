from prefect import serve
from upload_to_gcs import upload_to_gcs
from insurance_fraud_model import insurance_fraud_model

if __name__ == "__main__":
    upload_to_gcs_flow_deploy = upload_to_gcs.to_deployment(
        name='Upload Data to GCS',
        tags=['data', 'gcs']
    )
    model_pipe_flow_deploy = insurance_fraud_model.to_deployment(
        name='Train Insurance Fraud Model',
        tags=['model', 'balanced_rf_classifier']
    )
    serve(upload_to_gcs_flow_deploy, model_pipe_flow_deploy)