from prefect import serve
from prepare_data import upload_to_gcs
from orchestrate import insurance_fraud_model_pipe

if __name__ == "__main__":
    upload_to_gcs_flow_deploy = upload_to_gcs.to_deployment(
        name='Upload Data to GCS',
        tags=['data', 'gcs']
    )
    model_pipe_flow_deploy = insurance_fraud_model_pipe.to_deployment(
        name='Train Insurance Fraud Model',
        tags=['model', 'balanced_rf_classifier']
    )
    serve(upload_to_gcs_flow_deploy, model_pipe_flow_deploy)