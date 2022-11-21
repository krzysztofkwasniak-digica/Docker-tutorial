from pathlib import Path
from typing import Final, Union
import logging
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from src.logger import Logger
import os
DATASET_PATH: Final = Path("data/healthcare-dataset-stroke-data.csv")


class Objective:
    def __init__(self, X: pd.DataFrame, y: pd.DataFrame):
        self.X = X
        self.y = y
        self.best_acc: float = 0.0
        self.best_model = None

    def __call__(self, trial: optuna.trial.Trial) -> float:
        mlflow.end_run()
        with mlflow.start_run():
            X = np.copy(self.X)
            y = np.copy(self.y)

            params: dict[str, int] = dict(
                max_depth=trial.suggest_int("max_depth", 2, 50), n_estimators=trial.suggest_int("n_estimators", 50, 1000)
            )
            clf = RandomForestClassifier(n_jobs=-1, random_state=42, **params)
            mlflow.log_param("Classifier", "RandomForestClassifier")
            mlflow.log_params(params)

            kf = StratifiedKFold(n_splits=5)
            scores = []
            for train_index, test_index in kf.split(X, y):
                X_train = X[train_index]
                y_train = np.ravel(y[train_index])
                X_test = X[test_index]
                y_test = np.ravel(y[test_index])
                clf.fit(X_train, y_train)
                acc = clf.score(X_test, y_test)
                scores.append(acc)
            mean_acc = np.mean(scores)
            if mean_acc > self.best_acc:
                self.best_acc = mean_acc
                self.best_model = clf
                mlflow.sklearn.log_model(clf, "model")

        return mean_acc


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
    logger = Logger(__name__, log_level=logging.INFO)
    raw_data: pd.DataFrame = load_data(DATASET_PATH)

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        mlflow.create_experiment(name="Docker experiment")
    except mlflow.exceptions.RestException as e:
        logger.log.warning("Experiment already exists")
    mlflow.set_experiment(experiment_name="Docker experiment")

    X, y = preprocess_data(raw_data)
    objective = Objective(X, y)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)

    with mlflow.start_run(run_name="Final Results") as final_run:
        run_id: str = final_run.info.run_id
        model_registry_name: Final = "RandomForestClassifier"
        df = study.trials_dataframe()
        Path("temp").mkdir(exist_ok=True)
        df.to_csv("temp/results.csv")
        trial = study.best_trial
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("temp/optimization_history.png")
        mlflow.log_artifact("temp/optimization_history.png")
        mlflow.log_artifact("temp/results.csv")

        mlflow.log_param("Best Accuracy", trial.value)
        mlflow.log_params(trial.params)

        mlflow.sklearn.log_model(objective.best_model, "best_model")
        logger.log.info("Best trial params: ")
        for key, value in trial.params.items():
            logger.log.info(f"{key}: {value}")

        client = MlflowClient()
        try:
            client.create_registered_model(model_registry_name)
        except MlflowException:
            print("Model already registered")

        model_uri = "runs:/{}/best_model".format(run_id)
        client.create_model_version(model_registry_name, model_uri, run_id)


if __name__ == "__main__":
    main()
