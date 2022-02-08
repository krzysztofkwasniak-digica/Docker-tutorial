from functools import lru_cache

import mlflow
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from starlette.responses import RedirectResponse

from src.prediction_data import PredictionData


@lru_cache
def load_mlflow_client():
    mlflow.set_tracking_uri("sqlite:///db/store.db")
    client: MlflowClient = MlflowClient()
    return client


@lru_cache
def load_model(client: MlflowClient = Depends(load_mlflow_client)) -> RandomForestClassifier:
    model_source: str = client.get_latest_versions(name="RandomForestRegression")[-1].source
    model: RandomForestClassifier = mlflow.pyfunc.load_model(model_source)
    return model


app = FastAPI()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/predict")
def predict(prediction_data: PredictionData, model: RandomForestClassifier = Depends(load_model)) -> np.ndarray:
    pred_df: pd.DataFrame = pd.DataFrame(prediction_data.dict(), index=[0])
    prediction: np.uint8 = model.predict(pred_df)
    return {"Stroke": int(prediction)}
