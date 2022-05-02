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
from sklearn.metrics import make_scorer, accuracy_score, matthews_corrcoef, f1_score
from sklearn.model_selection import GridSearchCV
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
@click.option("--nested-cv", default=True, type=bool)
@click.option("-n", "--n-neighbors", default=5, type=int)
@click.option("--leaf-size", default=30, type=int)
@click.option("-c", "--c_reg", default=1, type=float)
@click.option("--max-iter", default=100, type=int)
@click.option("--n-estimators", default=100, type=int)
@click.option("--max-depth", default=None, type=int)
@click.option("-f", "--k-fold", default=5, type=int)
@click.option("--k-fold-inn", default=5, type=int)
@click.option("--shuffle", default=True, type=bool)
@click.option("-r", "--rand-state", default=42, type=int)
@click.option("-s", "--save-model-path", default="data/model.joblib",
                type=click.Path(dir_okay=False, writable=True, path_type=Path))
def train(ds_path: Path, preproc: str, n_features: int, model: str, nested_cv: bool, 
                n_neighbors: int, leaf_size: int, 
                c_reg: float, max_iter:int, 
                n_estimators: int, max_depth: int,
                k_fold:int, k_fold_inn:int, shuffle: bool, rand_state: int, 
                save_model_path: Path) -> None:
        
        data = pd.read_csv(ds_path)
        click.echo(f"Dataset shape: {data.shape}")

        X, y = data.drop(columns=["Id", "Cover_Type"]), data["Cover_Type"]

        classifier = create_pipeline(preproc, n_features, model, n_neighbors, leaf_size, c_reg, max_iter, 
                                        n_estimators, max_depth, rand_state)
        
        r_state = rand_state if shuffle == True else None
        cv_out = StratifiedKFold(n_splits=k_fold, shuffle=shuffle, random_state=r_state)
        out_res = {"accuracy": [], "matthews": [], "f1": []}
        with mlflow.start_run(run_name="_".join([model, "nested_k_fold"])):
                if nested_cv == True:
                        for train_ix, test_ix in cv_out.split(X):
                                X_train = X[train_ix, :]
                                X_test = X[test_ix, :]
                                y_train = y[train_ix]
                                y_test = y[test_ix]
                                with mlflow.start_run(run_name="_".join([model, "tuning_params"])):
                                        cv_in = StratifiedKFold(n_splits=k_fold_inn, shuffle=shuffle, random_state=r_state)
                                        params = {}
                                        if model == "knn":
                                                params = {"n_neighbors": range(1, 10),
                                                        "leaf_size": [10, 20, 30],
                                                        "weights": ["uniform", "distance"]
                                                }
                                        elif model == "logreg":
                                                params = {"C": [0.0001, 0.001, 0.05, 0.01, 0.5, 1],
                                                        "max_iter": [10, 50, 100, 150, 200],
                                                        "penalty": ["l1", "l2", "elasticnet", "none"]
                                                }
                                        elif model == "rf":
                                                params = {"n_estimators": [20, 50, 75, 100, 150],
                                                        "max_depth": [5, 7, 10, 20, 30, None],
                                                        "criterion": ["gini", "entropy"],
                                                        "max_features": ["sqrt", "log2", 0.9, 0.8, 0.7, 0.6] 
                                                }
                                        search = GridSearchCV(classifier, params,
                                                                scoring={"Matthews correlation": make_scorer(matthews_corrcoef), 
                                                                "Accuracy": "accuracy",
                                                                "F1": make_scorer(f1_score, average="weighted")},
                                                                cv=cv_in, refit=True, n_jobs=-1)
                                        results = search.fit(X_train, y_train)
                                        target = results.best_estimator_.predict(X_test)
                                        accuracy = accuracy_score(y_test, target)
                                        matthews = matthews_corrcoef(y_test, target)
                                        f1 = f1_score(y_test, target)
                                        out_res["accuracy"].append(accuracy)
                                        out_res["matthews"].append(matthews)
                                        out_res["f1"].append(f1)
                                        mlflow.sklearn.log_model(classifier, "".join(["forest_", results.best_estimator_]))
                                        for key, value in results.best_params_.items():
                                                mlflow.log_param(key, value)
                                        
                                        mlflow.log_metric("matthews_corrcoef", matthews)
                                        mlflow.log_metric("Accuracy score", accuracy)
                                        mlflow.log_metric("F1 score", f1)
                        mlflow.log_metric("matthews_corrcoef", np.mean(out_res["matthews"]))
                        mlflow.log_metric("Accuracy score", np.mean(out_res["accuracy"]))
                        mlflow.log_metric("F1 score", np.mean(out_res["f1"]))
                else:
                        cv_results = cross_validate(classifier, X, y, 
                                                scoring={"Matthews correlation": make_scorer(matthews_corrcoef), 
                                                        "Accuracy": "accuracy",
                                                        "F1": make_scorer(f1_score, average="weighted")}, 
                                                cv=kfold, return_estimator=True)

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

                mlflow.sklearn.log_model(classifier, "".join(["forest_", model]))
                mlflow.log_param("preprocess", preproc)
                mlflow.log_param("n_features", n_features)
                mlflow.log_param("k_fold", k_fold)
                mlflow.log_param("shuffle", shuffle)
                
        click.echo(f"Cross validation results: {cv_results}")
        dump(classifier, save_model_path)
        click.echo(f"Model has been saved to {save_model_path}")