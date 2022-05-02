import click
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef, f1_score
from joblib import dump
import mlflow
import mlflow.sklearn

@click.command()
@click.option("-p", "--ds-path", default="data/train.csv", 
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-n", "--n-neighbors", default=5, type=int)
@click.option("-f", "--k-fold", default=5, type=int)
@click.option("--shuffle", default=True, type=bool)
@click.option("-r", "--rand-state", default=42, type=int)
@click.option("-s", "--save-model-path", default="data/model.joblib",
                type=click.Path(dir_okay=False, writable=True, path_type=Path))
def train(ds_path: Path, n_neighbors: int, k_fold:int, shuffle: bool, rand_state: int, 
            save_model_path: Path) -> None:
    data = pd.read_csv(ds_path)
    click.echo(f"Dataset shape: {data.shape}")

    X, y = data.drop(columns=["Id", "Cover_Type"]), data["Cover_Type"]

    knn = KNeighborsClassifier(n_neighbors)
    with mlflow.start_run(run_name="KNN StratifiedKFold"):
        r_state = rand_state if shuffle == True else None
        kfold = StratifiedKFold(n_splits=k_fold, shuffle=shuffle, random_state=r_state)
        cv_results = cross_validate(knn, X, y, 
                                    scoring={"Matthews correlation": make_scorer(matthews_corrcoef), 
                                            "Accuracy": "accuracy",
                                            "F1": make_scorer(f1_score, average="weighted")}, 
                                    cv=kfold, return_estimator=True)
        mlflow.sklearn.log_model(knn, "forest_knn")
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("k_fold", k_fold)
        mlflow.log_param("shuffle", shuffle)
        mlflow.log_metric("matthews_corrcoef", np.mean(cv_results["test_Matthews correlation"]))
        mlflow.log_metric("Accuracy score", np.mean(cv_results["test_Accuracy"]))
        mlflow.log_metric("F1 score", np.mean(cv_results["test_F1"]))
        
        click.echo(f"Cross validation results: {cv_results}")
        dump(knn, save_model_path)
        click.echo(f"Model has been saved to {save_model_path}")