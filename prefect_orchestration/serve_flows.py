from prefect import serve
from upload_data_with_dummy_to_gcs import upload_data_with_dummy_to_gcs
from insurance_fraud_model import insurance_fraud_model
from raw_kaggle_to_gcs import raw_kaggle_to_gcs
from upload_data_no_dummy_to_gcs import upload_data_no_dummy_to_gcs

if __name__ == "__main__":
    raw_kaggle_to_gcs_deply = raw_kaggle_to_gcs.to_deployment(
        name='Upload Raw Kaggle data to GCS',
        tags=['kaggle', 'raw', 'gcs']
    )
    upload_data_no_dummy_to_gcs_deploy = upload_data_no_dummy_to_gcs.to_deployment(
        name='Upload Data with clean Cat Features to GCS',
        tags=['data', 'gcs', 'monitoring', 'cat']
    )
    upload_data_with_dummy_to_gcs_deploy = upload_data_with_dummy_to_gcs.to_deployment(
        name='Upload Data with Dummies to GCS',
        tags=['data', 'gcs']
    )
    model_pipe_flow_deploy = insurance_fraud_model.to_deployment(
        name='Train Insurance Fraud Model',
        tags=['model', 'balanced_rf_classifier']
    )

    serve(
        raw_kaggle_to_gcs_deply,
        upload_data_no_dummy_to_gcs_deploy,
        upload_data_with_dummy_to_gcs_deploy, 
        model_pipe_flow_deploy,
    )