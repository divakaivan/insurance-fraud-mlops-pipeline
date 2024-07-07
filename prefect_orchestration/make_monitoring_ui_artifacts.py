import os
import shap
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from prefect import task, flow
from prefect.flow_runs import pause_flow_run

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split

@task
def load_data_from_db() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load dataset with meaningful features, and same dataset but turned into dummies (dataset consists of only categorical features)"""
    
    db_params = {
        'host': os.getenv('db_host'),
        'port': os.getenv('db_port'), 
        'database': os.getenv('db_name'),
        'user': os.getenv('db_username'),
        'password': os.getenv('db_password')
    }
    engine = create_engine(f'postgresql+psycopg2://{db_params["user"]}:{db_params["password"]}@{db_params["host"]}:{db_params["port"]}/{db_params["database"]}')
    model_data_w_dummy = pd.read_sql_query('SELECT * FROM model_data_w_dummy', engine)
    meaningful_features_data = pd.read_sql_query('SELECT * FROM meaningful_features', engine)
    engine.dispose()

    return model_data_w_dummy, meaningful_features_data

@task()
def load_model_from_mlflow(model_mlflow_runid: str = None) -> object:
    """
        Load a model to be used
        If a run id is not provided, the env run id will be used
    """
    import mlflow
    tracking_uri = os.getenv('FRAUD_MODELLING_MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('Insurance Fraud Detection')
    if model_mlflow_runid is not None:
        run_id = model_mlflow_runid
    else:
        run_id = os.getenv('FRAUD_MODELLING_MLFLOW_RUN_ID')
    logged_model = f'runs:/{run_id}/balanced_rf_model'

    model = mlflow.pyfunc.load_model(logged_model)
    
    return model

@task
def prep_data_for_shap_graphs(model_data_w_dummy: pd.DataFrame) -> pd.DataFrame:
    """Prepare data for the next task in the flow"""
    
    X = model_data_w_dummy.drop('FraudFound_P', axis=1)

    return X

@task
def make_shap_graphs(model: object, X: pd.DataFrame) -> None:
    """Make SHAP graphs based on loaded model and data used for model training"""
    
    explainer = shap.Explainer(model)
    shap_values = explainer(X)

    shap.plots.waterfall(shap_values[0,:,0], max_display=10, show=False)
    plt.savefig('monitoring/shap_lime_info/not_fraud_shap_values.png')
    plt.close()

    shap.plots.waterfall(shap_values[0,:,1], max_display=10, show=False)
    plt.savefig('monitoring/shap_lime_info/fraud_shap_values.png')
    plt.close()

    mean_0 = np.mean(np.abs(shap_values.values[:, :, 0]), axis=0)
    mean_1 = np.mean(np.abs(shap_values.values[:, :, 1]), axis=0)

    df = pd.DataFrame({'Not Fraud': mean_0, 'Fraud': mean_1})

    _, ax = plt.subplots(1,1,figsize=(30, 20))
    df.plot.bar(ax=ax)

    ax.set_ylabel('Mean SHAP', size=20)
    ax.set_xticklabels(X.columns, rotation=90)
    ax.legend(fontsize=30)
    plt.savefig('monitoring/shap_lime_info/all_shap_values.png')
    plt.close()

    preds = model.predict(X)
    new_shap_values = []
    for i, pred in enumerate(preds):
        new_shap_values.append(shap_values.values[i][:, pred])

    shap_values.values = np.array(new_shap_values)
    shap.plots.bar(shap_values, max_display=10, show=False)
    plt.savefig('monitoring/shap_lime_info/mean_shap_values.png')
    plt.close()

    shap.plots.beeswarm(shap_values, max_display=10, show=False)
    plt.savefig('monitoring/shap_lime_info/beeswarm.png')
    plt.close()

    return None

@task
def prepare_data_for_evidently(model_data_w_dummy: pd.DataFrame, meaningful_features_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """I only have 1 set of data, so I split it to create reference/current data"""
    reference_data, current_data = train_test_split(model_data_w_dummy, test_size=0.2, random_state=42)

    with open('monitoring/model/balanced_rf_model/model.pkl', 'rb') as f:
        model = pickle.load(f)

    reference_data['prediction'] = model.predict(reference_data.drop(columns=['FraudFound_P']))
    current_data['prediction'] = model.predict(current_data.drop(columns=['FraudFound_P']))

    meaningful_reference_data, meaningful_current_data = train_test_split(meaningful_features_data, test_size=0.2, random_state=42)

    meaningful_reference_data['prediction'] = reference_data['prediction']
    meaningful_current_data['prediction'] = current_data['prediction']

    meaningful_reference_data.rename(columns={'FraudFound_P': 'target'}, inplace=True)
    meaningful_current_data.rename(columns={'FraudFound_P': 'target'}, inplace=True)

    return meaningful_reference_data, meaningful_current_data

@task
def make_evidently_html_dashboards(meaningful_reference_data: pd.DataFrame, meaningful_current_data: pd.DataFrame) -> None:
    """Create HTML evidently dashboards"""

    # Data tests dashboard
    from evidently.test_suite import TestSuite
    from evidently.test_preset import NoTargetPerformanceTestPreset
    from evidently.test_preset import DataQualityTestPreset
    from evidently.test_preset import DataStabilityTestPreset
    from evidently.test_preset import DataDriftTestPreset
    from evidently.test_preset import BinaryClassificationTestPreset

    data_stability = TestSuite(tests=[
        DataStabilityTestPreset(),
        NoTargetPerformanceTestPreset(),
        DataQualityTestPreset(),
        DataDriftTestPreset(),
        BinaryClassificationTestPreset(),
    ])

    data_stability.run(reference_data=meaningful_reference_data, current_data=meaningful_current_data)

    data_stability.save_html('monitoring/evidently_reports/data_stability.html')

    # Model prediction data dashboard
    from evidently.report import Report
    from evidently.metric_preset import DataQualityPreset, DataDriftPreset, ClassificationPreset, TargetDriftPreset

    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset(),
        ClassificationPreset(),
        TargetDriftPreset()
    ])

    report.run(reference_data=meaningful_reference_data, current_data=meaningful_current_data)

    report.save_html('monitoring/evidently_reports/evidently_dashboard.html')

    return None

@flow(log_prints=True)
def make_monitoring_ui_artifacts():
    """
        Update monitoring UI artifacts used by streamlit
        Default (from env) model run id is used, the user can input a new mlflow run id to use a new model
    """

    model_mlflow_runid = pause_flow_run(wait_for_input=str)

    model = load_model_from_mlflow(model_mlflow_runid)

    model_data_w_dummy, meaningful_features_data = load_data_from_db()
    X = prep_data_for_shap_graphs(model_data_w_dummy)
    make_shap_graphs(model, X)

    meaningful_reference_data, meaningful_current_data = prepare_data_for_evidently(model_data_w_dummy, meaningful_features_data)
    make_evidently_html_dashboards(meaningful_reference_data, meaningful_current_data)

if __name__ == "__main__":
    make_monitoring_ui_artifacts()