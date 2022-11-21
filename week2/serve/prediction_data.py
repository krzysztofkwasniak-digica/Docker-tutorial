from typing import Literal

from pydantic import BaseModel


class PredictionData(BaseModel):
    age: float = 27.00
    hypertension: Literal[0, 1] = 0
    heart_disease: Literal[0, 1] = 0
    avg_glucose_level: float = 219.84
    bmi: float = 64.40
    gender_Male: Literal[0, 1] = 0
    gender_Other: Literal[0, 1] = 0
    ever_married_Yes: Literal[0, 1] = 1
    Residence_type_Urban: Literal[0, 1] = 0
    smoking_status_never_smoked: Literal[0, 1] = 1
    smoking_status_smokes: Literal[0, 1] = 0
    work_type_Never_worked: Literal[0, 1] = 0
    work_type_Private: Literal[0, 1] = 1
    work_type_Self_employed: Literal[0, 1] = 0
    work_type_children: Literal[0, 1] = 0
