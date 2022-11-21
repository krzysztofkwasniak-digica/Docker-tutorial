FROM python:3.9-slim-buster

LABEL AUTHOR=krzysztof.kwasniak@digica.com

ARG PORT=8888
ENV PORT=$PORT

WORKDIR /app/

COPY serve/requirements.txt requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY serve/app.py app.py
COPY serve/prediction_data.py prediction_data.py
COPY serve/start_api.sh start_api.sh


EXPOSE $PORT

CMD [ "bash" ]

ENTRYPOINT [ "sh", "-c", "./start_api.sh $PORT" ]