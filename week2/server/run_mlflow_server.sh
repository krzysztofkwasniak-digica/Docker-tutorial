DB_DEST=${1:-db/store.db}
PORT=${2:-2137}
mlflow server \
    --default-artifact-root mlflow_artifacts \
    --backend-store-uri sqlite:///$DB_DEST \
    --host 0.0.0.0 \
    --port $PORT