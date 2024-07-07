import mlflow
import os

def get_best_params(run_id: str = os.getenv('FRAUD_MODELLING_MLFLOW_RUN_ID')) -> dict:
    """Get best parameters from the MLflow run"""
    tracking_uri = os.getenv('FRAUD_MODELLING_MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('Insurance Fraud Detection')
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    if params and next(iter(params)) == 'model_description':
        del params['model_description']
    return params

def convert_values_to_int_if_possible(dictionary: dict) -> dict:
    """Convert values in a dictionary to integers if possible"""
    converted_dict = {}
    for key, value in dictionary.items():
        try:
            converted_dict[key] = int(value)
        except ValueError:
            converted_dict[key] = value
    return converted_dict

def format_confusion_matrix(cm: list[list[int]]) -> str:
    labels = ['Actual Not Fraud', 'Actual Fraud']
    columns = ['Predicted Not Fraud', 'Predicted Fraud']

    md_table = "|  | " + " | ".join(columns) + " |\n"
    md_table += "|--------------------|-" + "-|".join(['---']*len(columns)) + "|\n"

    for i in range(len(labels)):
        md_table += f"| **{labels[i]}** | " + " | ".join([f"{cm[i][j]}" for j in range(len(columns))]) + " |\n"

    return md_table