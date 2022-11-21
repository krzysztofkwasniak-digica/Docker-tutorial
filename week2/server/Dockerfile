FROM python:3.9-slim-buster

ARG PORT=2137
ARG DB_DEST=db/store.db

ENV PORT=$PORT
ENV DB_DEST=$DB_DEST

COPY server/requirements.txt requirements.txt

RUN python -m pip install -r requirements.txt

COPY server/run_mlflow_server.sh run_mlflow_server.sh

EXPOSE $PORT

CMD [ "bash" ]

ENTRYPOINT [ "sh", "-c", "./run_mlflow_server.sh $DB_DEST $PORT" ]
