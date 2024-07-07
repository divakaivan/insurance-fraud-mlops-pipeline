from prefect import serve
from insurance_fraud_model import insurance_fraud_model
from raw_kaggle_to_gcs import raw_kaggle_to_gcs
from make_monitoring_ui_artifacts import make_monitoring_ui_artifacts

if __name__ == "__main__":
    raw_kaggle_to_gcs_deply = raw_kaggle_to_gcs.to_deployment(
        name='Upload Raw Kaggle data to GCS',
        tags=['kaggle', 'raw', 'gcs']
    )
    model_pipe_flow_deploy = insurance_fraud_model.to_deployment(
        name='Train Insurance Fraud Model',
        tags=['model', 'balanced_rf_classifier']
    )
    make_monitoring_ui_artifacts_deploy = make_monitoring_ui_artifacts.to_deployment(
        name='Create Monitoring UI Artifacts',
        tags=['monitoring', 'model', 'data', 'quality']
    )

    serve(
        raw_kaggle_to_gcs_deply,
        model_pipe_flow_deploy,
        make_monitoring_ui_artifacts_deploy,
    )