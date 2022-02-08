FROM python:3.9-slim-buster

LABEL AUTHOR=krzysztof.kwasniak@digica.com

WORKDIR /app/
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt && \
    python -m pip install -e .

COPY src/logger.py src/logger.py
COPY src/train.py src/train.py

ENTRYPOINT ["python", "src/train.py"]