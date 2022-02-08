FROM python:3.9-slim-buster

LABEL AUTHOR=krzysztof.kwasniak@digica.com

WORKDIR /app/

COPY requirements_app.txt requirements.txt
COPY setup.py setup.py
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e .

COPY src/prediction_data.py src/prediction_data.py
COPY src/app.py src/app.py
COPY start_api.sh start_api.sh

EXPOSE 8888

ENTRYPOINT ["bash", "start_api.sh"]