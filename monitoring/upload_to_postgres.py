from sqlalchemy import create_engine
import pandas as pd
import os
from google.cloud import storage

gcs_bucket_name = os.getenv('GCS_BUCKET_NAME')
meaningful_df_gcs_path = 'meaningful_df.csv'

db_user = os.getenv('POSTGRES_USER')
db_password = os.getenv('POSTGRES_PASSWORD')
db_host = 'db'  
db_port = '5432'
db_name = os.getenv('POSTGRES_DB')
table_name = os.getenv('POSTGRES_TABLE')

client = storage.Client()
bucket = client.get_bucket(gcs_bucket_name)
blob = bucket.blob(meaningful_df_gcs_path)

temp_csv_file = '/tmp/temp_csv_file.csv'
blob.download_to_filename(temp_csv_file)

df = pd.read_csv(temp_csv_file)

connection_string = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

engine = create_engine(connection_string)

try:
    df.to_sql(table_name, engine, if_exists='replace', index=False)
    print("Data inserted successfully")
except Exception as e:
    print(f"Error: {e}")
finally:
    if os.path.exists(temp_csv_file):
        os.remove(temp_csv_file)
