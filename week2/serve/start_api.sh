PORT=${1:-8889}
uvicorn app:app --reload --host "0.0.0.0" --port $PORT