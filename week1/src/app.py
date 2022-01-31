from functools import lru_cache
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI
from sklearn.ensemble import RandomForestClassifier
from starlette.responses import RedirectResponse

from src.prediction_data import PredictionData


@lru_cache
def load_model(model_path: Union[str, Path] = Path("models/model.joblib")) -> RandomForestClassifier:
    return joblib.load(model_path)


app = FastAPI()


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.post("/predict")
def predict(prediction_data: PredictionData, model: RandomForestClassifier = Depends(load_model)) -> np.ndarray:
    print(prediction_data.dict())
    pred_df: pd.DataFrame = pd.DataFrame(prediction_data.dict(), index=[0])
    prediction: np.uint8 = model.predict(pred_df)
    return {"Stroke": int(prediction)}
