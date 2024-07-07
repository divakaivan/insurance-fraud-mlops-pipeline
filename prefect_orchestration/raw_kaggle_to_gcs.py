import os

import kaggle
import pandas as pd

from prefect import task, flow
from prefect_gcp import GcpCredentials, GcsBucket


@task
def kaggle_to_local() -> str:
    """Downloads dataset from Kaggle and saves it to local"""

    dataset = "shivamb/vehicle-claim-fraud-detection"
    download_path = 'feature_n_model_exploration/raw_kaggle_data'
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    files_exist = any(os.scandir(download_path))

    if not files_exist:
        kaggle.api.dataset_download_files(dataset, path=download_path, unzip=True)
        print("Dataset downloaded")
    else:
        print(f"Dataset already exists in {os.path.abspath(download_path)}")

    return os.path.abspath(download_path)

@task
def local_to_gcs(file_path: str) -> None:
    df = pd.read_csv(file_path + '/fraud_oracle.csv')
    gcp_credentials = GcpCredentials.load("my-gcp-creds-block", validate=False)
    gcs_bucket = GcsBucket(
        bucket="fraud_modelling_prefect",
        gcp_credentials=gcp_credentials
    )

    gcs_bucket.upload_from_dataframe(df=df, to_path='raw_car_insurance_kaggle')
    
    return None

@flow(log_prints=True)
def raw_kaggle_to_gcs():
    file_path = kaggle_to_local()
    local_to_gcs(file_path)

if __name__ == '__main__':
    raw_kaggle_to_gcs()