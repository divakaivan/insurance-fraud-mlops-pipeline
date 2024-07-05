from pathlib import Path
from prefect import flow
from prefect_gcp import GcpCredentials, GcsBucket
import pandas as pd

@flow
def upload_to_gcs():
    name = 'ready_df.csv'
    file_path = Path(name)
    df = pd.read_csv(file_path)
    gcp_credentials = GcpCredentials.load("my-gcp-creds-block", validate=False)
    gcs_bucket = GcsBucket(
        bucket="fraud_modelling_prefect",
        gcp_credentials=gcp_credentials
    )

    gcs_bucket.upload_from_dataframe(df=df, to_path=name)
    
    return None

if __name__ == "__main__":
    upload_to_gcs.serve(name="upload_to_gcs")