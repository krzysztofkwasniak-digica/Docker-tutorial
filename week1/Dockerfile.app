FROM

WORKDIR
COPY setup.py setup.py
RUN python -m pip install -e .
# It has to match the port that the app will start on.
EXPOSE 8888
ENTRYPOINT
