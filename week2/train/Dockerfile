FROM python:3.9-slim-buster

LABEL AUTHOR=krzysztof.kwasniak@digica.com

WORKDIR /app/

COPY train/requirements.txt requirements.txt

RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY train/logger.py logger.py
COPY train/train.py train.py

ENTRYPOINT ["python", "src/train.py"]