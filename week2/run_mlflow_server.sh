mlflow server \
    --default-artifact-root mlflow_artifacts \
    --backend-store-uri sqlite:///db/store.db \
    --host 0.0.0.0 \
    --port 2137