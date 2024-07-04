from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from prefect import flow, task
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from utils import get_best_params, convert_values_to_int_if_possible
from prefect.artifacts import create_markdown_artifact, create_table_artifact
import mlflow

import mlflow
import mlflow.sklearn
import os
from dotenv import load_dotenv

load_dotenv()

def format_confusion_matrix(cm):
    labels = ['Actual Not Fraud', 'Actual Fraud']
    columns = ['Predicted Not Fraud', 'Predicted Fraud']

    md_table = "|  | " + " | ".join(columns) + " |\n"
    md_table += "|--------------------|-" + "-|".join(['---']*len(columns)) + "|\n"

    for i in range(len(labels)):
        md_table += f"| **{labels[i]}** | " + " | ".join([f"{cm[i][j]}" for j in range(len(columns))]) + " |\n"

    return md_table

@task
def read_data(filename: str) -> pd.DataFrame:
    return pd.read_csv(filename)

@task
def split_data(data: pd.DataFrame, test_size: float = 0.2) -> tuple:
        
    dummies_X = data.drop(columns=['FraudFound_P'])
    y = data['FraudFound_P']
    X_train, X_test, y_train, y_test = train_test_split(dummies_X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

@task
def best_params_from_mlflow():
    best_params = get_best_params()
    best_params = convert_values_to_int_if_possible(best_params)
    return best_params

@task
def train_model(X_train: pd.DataFrame, y_train: pd.Series, X_test, y_test) -> None:
    
    with mlflow.start_run():

        best_params = best_params_from_mlflow()
        mlflow.log_params(best_params)
        model = BalancedRandomForestClassifier(**best_params)
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(model, 'balanced_rf_model')

        y_pred = model.predict(X_test)
        
        mlflow.log_metric('recall', recall_score(y_test, y_pred))
        cr = classification_report(y_test, y_pred, output_dict=True, target_names=['Not Fraud', 'Fraud'])
        mlflow.log_dict(cr, './classification_report.json')
        md_classification_report = f"""
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
"""
        for cls in ['Not Fraud', 'Fraud']:
            md_classification_report += f"| {cls} | "
            md_classification_report += f"{cr[cls]['precision']:.4f} | "
            md_classification_report += f"{cr[cls]['recall']:.4f} | "
            md_classification_report += f"{cr[cls]['f1-score']:.4f} | "
            md_classification_report += f"{cr[cls]['support']:.1f} |\n"
        cm = confusion_matrix(y_test, y_pred)
        mlflow.log_figure(ConfusionMatrixDisplay(cm).plot().figure_, './confusion_matrix.png')
        md_confussion_matrix = f"""
{format_confusion_matrix(cm)}
"""
        create_markdown_artifact(
            key="modelreport",
            markdown=md_classification_report+md_confussion_matrix
        )

    return None

@flow(log_prints=True)
def main():

    # MLflow settings
    # os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '../my-creds.json'
    tracking_uri = os.getenv('FRAUD_MODELLING_MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('Insurance Fraud Detection')

    # Load
    data = read_data("./ready_df.csv")
    
    # Split
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Train
    train_model(X_train, y_train, X_test, y_test)
    
if __name__ == "__main__":
    main()