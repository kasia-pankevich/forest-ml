import click
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef, f1_score
from joblib import dump
import mlflow
import mlflow.sklearn

def create_pipeline(preprocess: str, n_features: int, model: str, n_neighbors: int, leaf_size: int, 
                        c_reg: float, max_iter:int, n_estimators: int, max_depth: int, rand_state: int) -> Pipeline:
        preproc_map = {
                "none": None, 
                "min_max": MinMaxScaler(), 
                "standard": StandardScaler(),
                "pca" : PCA(n_features), 
                "svd": TruncatedSVD(n_features)
        }
        model_map = {
                "knn":
                KNeighborsClassifier(n_neighbors, leaf_size=leaf_size, 
                                        weights="distance", n_jobs=-1),
                "logreg":
                LogisticRegression(C=c_reg, max_iter=max_iter, 
                                        random_state=rand_state, n_jobs=-1),
                "rf": RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                        random_state=rand_state, n_jobs=-1)
        }
        preprocessor = preproc_map[preprocess.lower()]
        ppl_steps = []
        if preprocessor is not None:
                ppl_steps.append(("preprocessing", preprocessor))
        classifier = model_map[model.lower()]
        ppl_steps.append(("classification", classifier))
        return Pipeline(ppl_steps)


@click.command()
@click.option("-p", "--ds-path", default="data/train.csv", 
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--preproc", default="none", 
                type=click.Choice(["none", "min_max", "standard", "pca", "svd"], case_sensitive=False))
@click.option("--n-features", default=3, type=int)
@click.option("-m", "--model", default="knn", type=click.Choice(["knn", "logreg", "rf"], case_sensitive=False))
@click.option("-n", "--n-neighbors", default=5, type=int)
@click.option("--leaf-size", default=30, type=int)
@click.option("-c", "--c_reg", default=1, type=float)
@click.option("--max-iter", default=100, type=int)
@click.option("--n-estimators", default=100, type=int)
@click.option("--max-depth", default=None, type=int)
@click.option("-f", "--k-fold", default=5, type=int)
@click.option("--shuffle", default=True, type=bool)
@click.option("-r", "--rand-state", default=42, type=int)
@click.option("-s", "--save-model-path", default="data/model.joblib",
                type=click.Path(dir_okay=False, writable=True, path_type=Path))
def train(ds_path: Path, preproc: str, n_features: int, model: str, 
                n_neighbors: int, leaf_size: int, 
                c_reg: float, max_iter:int, 
                n_estimators: int, max_depth: int,
                k_fold:int, shuffle: bool, rand_state: int, 
                save_model_path: Path) -> None:
        
        data = pd.read_csv(ds_path)
        click.echo(f"Dataset shape: {data.shape}")

        X, y = data.drop(columns=["Id", "Cover_Type"]), data["Cover_Type"]

        classifier = create_pipeline(preproc, n_features, model, n_neighbors, leaf_size, c_reg, max_iter, 
                                        n_estimators, max_depth, rand_state)
        
        with mlflow.start_run(run_name=" ".join([model, "StratifiedKFold"])):
                r_state = rand_state if shuffle == True else None
                kfold = StratifiedKFold(n_splits=k_fold, shuffle=shuffle, random_state=r_state)
                cv_results = cross_validate(classifier, X, y, 
                                        scoring={"Matthews correlation": make_scorer(matthews_corrcoef), 
                                                "Accuracy": "accuracy",
                                                "F1": make_scorer(f1_score, average="weighted")}, 
                                        cv=kfold, return_estimator=True)

                mlflow.sklearn.log_model(classifier, "".join(["forest_", model]))
                mlflow.log_param("preprocess", preproc)
                mlflow.log_param("n_features", n_features)
                mlflow.log_param("k_fold", k_fold)
                mlflow.log_param("shuffle", shuffle)

                if model == "knn":
                        mlflow.log_param("n_neighbors", n_neighbors)
                        mlflow.log_param("leaf_size", leaf_size)
                elif model == "logreg":
                        mlflow.log_param("C", c_reg)
                        mlflow.log_param("max_iter", max_iter)
                elif model == "rf":
                        mlflow.log_param("n_estimators", n_estimators)
                        mlflow.log_param("max_depth", max_depth)
                
                mlflow.log_metric("matthews_corrcoef", np.mean(cv_results["test_Matthews correlation"]))
                mlflow.log_metric("Accuracy score", np.mean(cv_results["test_Accuracy"]))
                mlflow.log_metric("F1 score", np.mean(cv_results["test_F1"]))
                
        click.echo(f"Cross validation results: {cv_results}")
        dump(classifier, save_model_path)
        click.echo(f"Model has been saved to {save_model_path}")