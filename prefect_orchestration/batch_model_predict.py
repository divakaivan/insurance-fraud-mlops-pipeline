import os
import pandas as pd
from sqlalchemy import create_engine
from prefect import flow, task
from prefect.flow_runs import pause_flow_run
from make_monitoring_ui_artifacts import load_model_from_mlflow

@task
def read_new_data(table_name: str) -> pd.DataFrame:
    """Reads data to be predicted from the database"""
    db_params = {
        'host': os.getenv('db_host'),
        'port': os.getenv('db_port'), 
        'database': os.getenv('db_name'),
        'user': os.getenv('db_username'),
        'password': os.getenv('db_password')
    }
    engine = create_engine(f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')
    query = f"SELECT * FROM {table_name}"
    data = pd.read_sql_query(query, engine)
    engine.dispose()

    return data

@task
def predict(model: object, data: pd.DataFrame, get_probs: bool = False) -> pd.DataFrame:
    """
    Predicts on the data using the model provided. 
    If get_probs is True, it also returns the probabilities of the predictions.
    """
    y_pred = model.predict(data)

    data_with_predictions = data.copy()
    data_with_predictions['FraudFound_P'] = y_pred

    if get_probs:
        y_probs = model.predict_proba(data)
        data_with_predictions['FraudFound_P_Prob'] = y_probs

    return data_with_predictions

@task
def save_predictions(data: pd.DataFrame, table_name: str) -> None:
    """Saves the predictions to the database"""

    db_params = {
        'host': os.getenv('db_host'),
        'port': os.getenv('db_port'), 
        'database': os.getenv('db_name'),
        'user': os.getenv('db_username'),
        'password': os.getenv('db_password')
    }
    engine = create_engine(f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')
    data.to_sql(table_name, engine, if_exists='append', index=False)
    engine.dispose()

@flow(log_prints=True)
def batch_model_predict():
    """Orchestration flow to predict on new data and save the predictions to the database"""
    table_name = pause_flow_run(wait_for_input=str)
    data = read_new_data(table_name)
    model = load_model_from_mlflow()
    data_with_predictions = predict(model, data)
    save_predictions(data_with_predictions, 'model_data_w_predictions')

if __name__ == '__main__':
    batch_model_predict()
