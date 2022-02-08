FROM python:3.9-slim-buster

COPY requirements_mlflow_server.txt requirements.txt

RUN python -m pip install -r requirements.txt

COPY run_mlflow_server.sh run_mlflow_server.sh

EXPOSE 8888

ENTRYPOINT ["bash", "./run_mlflow_server.sh"]
