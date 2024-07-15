# Machine Learning Canvas

![image](https://github.com/user-attachments/assets/b6b1dc7c-d518-4ee8-b304-df7db538b12b)

Template from [ownml.co](https://www.ownml.co/)

# Project diagram

![Project diagram](/project_info/project_diagram.png)

- **Terraform** is used to provision Google Cloud Storage (GCS) buckets for data storage and a virtual machine for MLflow
- **Kaggle** [data](https://www.kaggle.com/datasets/shivamb/vehicle-claim-fraud-detection) related to Car Insurance Fraud claims is used
- **Google Cloud Storage** is used as a data lake to store the raw Kaggle data
- **PostgreSQL** (and **pgAdmin**) is used as a data mart to store processed data, and data ready for model training
- **Prefect** and **Prefect Cloud** is used for orchestration. Flows used:
  - upload raw Kaggle data to GCS
  - train a new model
  - do batch model prediction
  - update Evidently monitoring artifacts (HTMLs + PNGs)
- **MLflow** is used for experiment tracking. It is hosted on a GCP VM
- **Docker** is used to run Grafana, PostgreSQL, pgAdmin, and FastAPI (for optional web-service deployment)
- **Grafana** is used to create a dashboard using data from the postgres database
- **Evidently** is used to create data (to compare current vs reference data distributions) and monitoring html reports which are hosted using [Streamlit](https://insurance-fraud-model-monitoring.streamlit.app/)
- Other features:
  - [Python documentation](https://fraud-model-prefect-docs.netlify.app/) using sphinx
  - Makefile for easy setup and start
  - git pre-commit hooks
  - pytests

# Feature engineering

- Target variable is FraudFound_P - 0 for Not Fraud, and 1 for Fraud
- **Information Value (IV)** and **Weight of Evidence (WoE)** were used for feature engineering ([feature_n_model_exploration/feature_eng.ipynb](/feature_n_model_exploration/feature_eng.ipynb)) and this achieved a feature count decrease from 32 to 13

# Model selection

- Selection criteria: highest recall. _Reasoning:_ recognise the most Fraud cases
- The best model ended up being a **Balanced Random Forest Classifier**, achieving 92-96% recall
- Code in [feature_n_model_exploration/experiment_tracking.ipynb](/feature_n_model_exploration/experiment_tracking.ipynb)

# Monitoring

- Evidently dashboard ([public link](https://insurance-fraud-model-monitoring.streamlit.app/)). It includes:
  - reference vs current dataset comparison dashboard
  - data test dashboard
  - SHAP values

- Grafana dashboard
![Grafana dashboard](/project_info/grafana_dashboard.png)

### Reproducability

- clone the repository
```
https://github.com/divakaivan/insurance-fraud-mlops-pipeline.git
```
- add environment variables and rename `sample.env` to `.env`. This is used for running the Docker services
- type `make` in the terminal and you should see something similar to set up the project. You need to have a Prefect Cloud account and already logged in to serve the prefect flows
```
Usage: make [option]

Options:
  help                 Show this help message
  gcp-setup            View GCP resources to be created (buckets for mlflow artifacts, raw data, and start a VM that runs mlflow on start)
  gcp-create           Create GCP resources 
  prefect-serve-cloud  Serve Model Train and Load to GCS flows to Prefect Cloud
  build-all            Build image with PostgreSQL, pgAdmin, Grafana, Data upload to db, FastAPI
  start-all            Start services
  monitoring           Update monitoring artifacts 

NOTE! If you update the monitoring UI artifacts, you have to push them to GitHub to update the hosted UI

prefect-serve-cloud and start-all are not run in detached mode
```

### Things to consider for improvements

- try more model ensambles and sampling algorithms (i.e. [ADASYN](https://ieeexplore.ieee.org/document/4633969))
- adding more (data/code/model stress) tests
- simulating a new dataset and using it on the batch predict flow
- more complex monitoring dashboards using Grafana and Evidently

### Blog posts about this project

I developed the current version (as of 8th of July 2024) project in the span of 8 days and discussed each day in my self-study blog:

- [Day 182](https://50daysml.blogspot.com/2024/07/day-182-learning-about-feature.html): Learning about feature selection in fraud detection and finding a classifier model with low recall
- [Day 183](https://50daysml.blogspot.com/2024/07/day-183-failing-to-install-kubeflow-and.html): Failing to install Kubeflow, and setting up mlflow on GCP
- [Day 184](https://50daysml.blogspot.com/2024/07/day-184-mlflow-experiment-tracking-and.html): Mlflow experiment tracking and trying out metaflow
- [Day 185](https://50daysml.blogspot.com/2024/07/day-185-using-prefect-as-my.html): Using prefect as my orchestrator for my MLOps project
- [Day 186](https://50daysml.blogspot.com/2024/07/day-186-prefect-cloud-model-serving.html): Prefect cloud, model serving with FastAPI, and SHAP values
- [Day 187](https://50daysml.blogspot.com/2024/07/day-187-setting-up-postgres-pgadmin.html): Setting up postgres, pgAdmin, Grafana and FastAPI to run in Docker
- [Day 188](https://50daysml.blogspot.com/2024/07/day-188-setting-up-automatically.html): Setting up automatically updated monitoring UI using streamlit
- [Day 189](https://50daysml.blogspot.com/2024/07/day-189-i-finished-car-insurance-fraud.html): I finished the Car Insurance Fraud MLOps project. Thank you MLOps zoomcamp for teaching me so much!
