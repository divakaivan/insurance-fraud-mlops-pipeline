```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri sqlite:///fraud_modelling.sqlite --default-artifact-root gs://fraud_modelling_artifacts
```