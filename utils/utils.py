import mlflow
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, ConfusionMatrixDisplay, recall_score
import os

def save_feature_importance(model, model_name, X_test):
    if model_name == 'LogisticRegression':
        coefs = model.coef_[0].tolist() 
    else:
        coefs = model.feature_importances_.tolist()
    
    coefs = [float(value) for value in coefs]
    features = X_test.columns.tolist()
    feature_importance_dict = dict(zip(features, coefs))
    
    feature_importance_json = json.dumps(feature_importance_dict, indent=4)
    json_file_path = f"{model_name}_feature_importance.json"
    with open(json_file_path, "w") as json_file:
        json_file.write(feature_importance_json)
    
    mlflow.log_artifact(json_file_path, "model_report/")
    
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': coefs})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    feature_importance_png_path = f"{model_name}_feature_importance.png"
    plt.savefig(feature_importance_png_path)
    plt.close()
    mlflow.log_artifact(feature_importance_png_path, "model_report/")

def log_model_performance(model, X_test, y_test, model_name, model_description, best_params=None):
    with mlflow.start_run():
        mlflow.set_tag('model_name', model_name)
        mlflow.log_param('model_description', model_description)
        mlflow.sklearn.log_model(model, model_name)
        
        y_pred = model.predict(X_test)
        mlflow.log_metric('recall', recall_score(y_test, y_pred))
        
        save_feature_importance(model, model_name, X_test)
        
        classification_rep = classification_report(y_test, y_pred)
        mlflow.log_text(classification_rep, f"model_report/{model_name}_classification_report.txt")
        print("Classification Report:")
        print(classification_rep)
        
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues', values_format='d')
        plt.title('Confusion Matrix')
        plt.grid(False)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        cm_img_path = f"{model_name}_confusion_matrix.png"
        plt.savefig(cm_img_path)
        plt.close()
        mlflow.log_artifact(cm_img_path, f"model_report/")
        
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        print("ROC AUC Score:", roc_auc)
        mlflow.log_metric("roc_auc_score", roc_auc)
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        plt.figure()
        plt.plot(fpr, tpr, color='orange', label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_curve_path = f"{model_name}_roc_curve.png"
        plt.savefig(roc_curve_path)
        plt.close()
        mlflow.log_artifact(roc_curve_path, f"model_report/")
        
        if best_params is not None:
            mlflow.log_params(best_params)

def get_best_params():
    tracking_uri = os.getenv('FRAUD_MODELLING_MLFLOW_TRACKING_URI')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment('Insurance Fraud Detection')
    run_id = os.getenv('FRAUD_MODELLING_MLFLOW_RUN_ID')
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    params = run.data.params
    if params and next(iter(params)) == 'model_description':
        del params['model_description']
    return params
