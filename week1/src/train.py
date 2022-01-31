from pathlib import Path
from typing import Final, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.logger import Logger

DATASET_PATH: Final = Path("data/healthcare-dataset-stroke-data.csv")


def load_data(data_path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(data_path)


def preprocess_data(raw_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_no_idx: pd.DataFrame = raw_data.drop(columns=["id"])  # Drop Id column as it holds no useful information
    data_drop_unknown: pd.DataFrame = data_no_idx[data_no_idx["smoking_status"] != "Unknown"]
    data_no_na: pd.DataFrame = data_drop_unknown[data_drop_unknown["bmi"] != "N/A"].dropna()
    data_dummy: pd.DataFrame = pd.get_dummies(
        data_no_na,
        columns=["gender", "ever_married", "Residence_type", "smoking_status", "work_type"],
        drop_first=True,
    )
    data_dummy = data_dummy.rename(
        columns={
            "work_type_Self-employed": "work_type_Self_employed",
            "smoking_status_never smoked": "smoking_status_never_smoked",
        }
    )
    X: pd.DataFrame = data_dummy.drop(columns=["stroke"])
    y: pd.DataFrame = data_dummy["stroke"]
    return X, y


def main() -> None:
    logger = Logger(__name__)
    raw_data: pd.DataFrame = load_data(DATASET_PATH)
    X, y = preprocess_data(raw_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    results: float = clf.score(X_test, y_test)
    logger.log.info(f"Accuracy: {results}")
    model_dir: Path = Path("models")
    model_dir.mkdir(exist_ok=True)
    model_save_path: Path = model_dir / "model.joblib"
    joblib.dump(clf, model_save_path)
    logger.log.info(f"Model successfully saved to {model_save_path}")


if __name__ == "__main__":
    main()
