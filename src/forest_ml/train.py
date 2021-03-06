import click
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, KFold, StratifiedKFold
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.metrics import matthews_corrcoef, f1_score
from sklearn.model_selection import GridSearchCV
from joblib import dump
import mlflow
import mlflow.sklearn
import sys
import warnings
import os
import forest_ml.features_preparing as fp
from typing import Dict, List, Any, Optional


def try_convert_to_number_tryexcept(s):
    # Returns number if string is a number, otherwise return initial string
    try:
        dig = float(s)
        return dig
    except ValueError:
        return s


def create_pipeline(
    preprocess: str,
    n_features: int,
    model: str,
    n_neighbors: int,
    leaf_size: int,
    c_reg: float,
    max_iter: int,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    min_samples_split: int,
    criterion: str,
    max_features: object,
    rand_state: int,
) -> Pipeline:
    preproc_map = {
        "none": None,
        "min_max": MinMaxScaler(),
        "standard": StandardScaler(),
        "pca": PCA(n_features),
        "svd": TruncatedSVD(n_features),
    }

    if max_features.isdigit() is True:
        max_features = int(max_features)
    else:
        max_features = try_convert_to_number_tryexcept(max_features)

    model_map = {
        "knn": KNeighborsClassifier(
            n_neighbors, leaf_size=leaf_size, weights="distance", n_jobs=-1
        ),
        "logreg": LogisticRegression(
            C=c_reg,
            max_iter=max_iter,
            solver="saga",
            random_state=rand_state,
            n_jobs=-1,
        ),
        "rf": RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=rand_state,
            n_jobs=-1,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        ),
        "et": ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=rand_state,
            n_jobs=-1,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
        ),
    }
    preprocessor = preproc_map[preprocess.lower()]
    ppl_steps = []
    if preprocessor is not None:
        ppl_steps.append(("preprocessing", preprocessor))
    classifier = model_map[model.lower()]
    ppl_steps.append(("clf", classifier))
    return Pipeline(ppl_steps)


def train_with_nested_cv(
    classifier: Pipeline,
    X: pd.DataFrame,
    y: pd.DataFrame,
    cv_out: KFold,
    model: str,
    k_fold_inn: int,
    shuffle: bool,
    rand_state: Optional[int],
) -> Dict[str, List[float]]:
    out_res: Dict[str, List[float]] = {
        "accuracy": [],
        "matthews": [],
        "f1": [],
    }
    for train_ix, test_ix in cv_out.split(X, y):
        sample_size = 7000
        mlflow.log_param("sample_size", sample_size)
        X_train = X[train_ix[:sample_size], :]
        X_test = X[test_ix, :]
        y_train = y[train_ix[:sample_size]]
        y_test = y[test_ix]
        with mlflow.start_run(
            nested=True, run_name="_".join([model, "tuning_params"])
        ):
            cv_in = StratifiedKFold(
                n_splits=k_fold_inn, shuffle=shuffle, random_state=rand_state
            )
            params = {}
            if model == "knn":
                params = {
                    "clf__n_neighbors": range(1, 10),
                    "clf__leaf_size": [10, 20, 30],
                    "clf__weights": ["uniform", "distance"],
                }
            elif model == "logreg":
                params = {
                    "clf__C": [0.001, 0.01, 0.5, 1, 10, 100, 1000],
                    "clf__max_iter": [10, 50, 100, 150, 200],
                    "clf__penalty": ["l1", "l2", "elasticnet"],
                }
            elif model in ["rf", "et"]:
                params = {
                    "clf__n_estimators": [100, 200, 300, 400, 500],
                    # "clf__max_depth": [10, 15, 20, 25, 30, None],
                    "clf__min_samples_split": [2, 3, 5, 7, 9],
                    "clf__min_samples_leaf": [1, 2, 4, 6, 8],
                    "clf__criterion": ["gini", "entropy"],
                    "clf__max_features": [
                        "auto",
                        "sqrt",
                        "log2",
                        0.5,
                        0.4,
                        0.3,
                        None,
                    ],
                }
            search = GridSearchCV(
                classifier,
                params,
                scoring={
                    "Matthews correlation": make_scorer(matthews_corrcoef),
                    "Accuracy": "accuracy",
                    "F1": make_scorer(f1_score, average="weighted"),
                },
                cv=cv_in,
                refit="Accuracy",
                n_jobs=-1,
            )
            results = search.fit(X_train, y_train)
            target = results.best_estimator_.predict(X_test)
            accuracy = accuracy_score(y_test, target)
            matthews = matthews_corrcoef(y_test, target)
            f1 = f1_score(y_test, target, average="weighted")
            out_res["best_params"] = results.best_params_
            out_res["accuracy"].append(accuracy)
            out_res["matthews"].append(matthews)
            out_res["f1"].append(f1)
            mlflow.sklearn.log_model(
                results.best_estimator_, "".join(["forest_", model])
            )
            for key, value in results.best_params_.items():
                mlflow.log_param(key, value)

            mlflow.log_metric("matthews_corrcoef", matthews)
            mlflow.log_metric("Accuracy score", accuracy)
            mlflow.log_metric("F1 score", f1)
            classifier = results.best_estimator_
    mlflow.log_metric("matthews_corrcoef", np.mean(out_res["matthews"]))
    mlflow.log_metric("Accuracy score", np.mean(out_res["accuracy"]))
    mlflow.log_metric("F1 score", np.mean(out_res["f1"]))
    return out_res


def train_without_nested_cv(
    classifier: Pipeline,
    X: pd.DataFrame,
    y: pd.DataFrame,
    kfold: KFold,
    model: str,
    n_neighbors: int,
    leaf_size: int,
    c_reg: float,
    max_iter: int,
    n_estimators: int,
    max_depth: int,
    criterion: str,
    max_features: object,
    k_fold: int,
) -> Any:
    cv_results = cross_validate(
        classifier,
        X,
        y,
        scoring={
            "Matthews correlation": make_scorer(matthews_corrcoef),
            "Accuracy": "accuracy",
            "F1": make_scorer(f1_score, average="weighted"),
        },
        cv=kfold,
        return_estimator=True,
    )

    if model == "knn":
        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("leaf_size", leaf_size)
    elif model == "logreg":
        mlflow.log_param("C", c_reg)
        mlflow.log_param("max_iter", max_iter)
    elif model in ["rf", "et"]:
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_features", max_features)

    mlflow.sklearn.log_model(classifier, "".join(["forest_", model]))
    mlflow.log_metric(
        "matthews_corrcoef", np.mean(cv_results["test_Matthews correlation"])
    )
    mlflow.log_metric("Accuracy score", np.mean(cv_results["test_Accuracy"]))
    mlflow.log_metric("F1 score", np.mean(cv_results["test_F1"]))
    return cv_results


@click.command()
@click.option(
    "-d",
    "--ds-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to train dataset csv",
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Local folder path and file name to save model to",
)
@click.option(
    "--preproc",
    default="none",
    type=click.Choice(
        ["none", "min_max", "standard", "pca", "svd"], case_sensitive=False
    ),
    help="Data preprocessing type",
)
@click.option(
    "--n-features",
    default=3,
    type=int,
    help="Number of features to leave for dimensionality reduction."
    + "Used if preproc set to 'pca' or 'svd'",
)
@click.option(
    "-m",
    "--model",
    default="et",
    type=click.Choice(["knn", "logreg", "rf", "et"], case_sensitive=False),
    help="Name of model to train: knn for K-Nearest Neighbors,"
    + "logreg for Logistic Regression, rf for Random Forest Classifier,"
    + "et for Extra Trees Classifier",
)
@click.option(
    "--nested-cv",
    default=False,
    type=bool,
    help="Wether to do model hyperparameters tuning"
    + "(through nested cross-validation)",
)
@click.option(
    "-n",
    "--n-neighbors",
    default=5,
    type=int,
    help="Hyperparameter for KNN classifier",
)
@click.option(
    "--leaf-size",
    default=30,
    type=int,
    help="Hyperparameter for KNN classifier",
)
@click.option(
    "-c",
    "--c_reg",
    default=1,
    type=float,
    help="Hyperparameter for Logistic regression classifier",
)
@click.option(
    "--max-iter",
    default=100,
    type=int,
    help="Hyperparameter for Logistic regression classifier",
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    help="Hyperparameter for Random Forest or Extra Trees classifier",
)
@click.option(
    "--criterion",
    default="gini",
    type=click.Choice(["gini", "entropy"]),
    help="Hyperparameter for Random forest or Extra Trees classifier",
)
@click.option(
    "--max-features",
    default="auto",
    help="Hyperparameter for Random forest or Extra Trees classifier."
    + "Values: {'auto', 'sqrt', 'log2', int or float}",
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    help="Hyperparameter for Random forest or Extra Trees classifier",
)
@click.option(
    "--min-samples-leaf",
    default=1,
    type=int,
    help="Hyperparameter for Random forest or Extra Trees classifier",
)
@click.option(
    "--min-samples-split",
    default=2,
    type=int,
    help="Hyperparameter for Random forest or Extra Trees classifier",
)
@click.option(
    "-f",
    "--k-fold",
    default=5,
    type=int,
    help="Number of folds for k-fold prediction evaluation",
)
@click.option(
    "--k-fold-inn",
    default=3,
    type=int,
    help="Number of folders for k-fold hyperparameters tuning"
    + "(used in nested CV)",
)
@click.option(
    "--shuffle",
    default=True,
    type=bool,
    help="Wether to shuffle data during splitting",
)
@click.option(
    "-r",
    "--rand-state",
    default=42,
    type=int,
    help="Random state to fix experiments results",
)
def train(
    ds_path: Path,
    preproc: str,
    n_features: int,
    model: str,
    nested_cv: bool,
    n_neighbors: int,
    leaf_size: int,
    c_reg: float,
    max_iter: int,
    n_estimators: int,
    max_depth: int,
    criterion: str,
    max_features: object,
    min_samples_leaf: int,
    min_samples_split: int,
    k_fold: int,
    k_fold_inn: int,
    shuffle: bool,
    rand_state: int,
    save_model_path: Path,
) -> None:

    if model == "logreg" and not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore"  # Also affect subprocesses

    data = pd.read_csv(ds_path)

    click.echo(f"Dataset shape: {data.shape}")

    X, y = data.drop(columns=["Cover_Type"]), data["Cover_Type"]
    X = fp.prepare(X).to_numpy()
    click.echo(f"Number of features: {X.shape[1]}")
    y = y.to_numpy()

    if n_features < 1 or n_features > X.shape[1]:
        raise ValueError(
            "Invalid value for '--n-features'."
            + f"It should be in a range [1, {X.shape[1]}]"
        )

    classifier = create_pipeline(
        preproc,
        n_features,
        model,
        n_neighbors,
        leaf_size,
        c_reg,
        max_iter,
        n_estimators,
        max_depth,
        min_samples_leaf,
        min_samples_split,
        criterion,
        max_features,
        rand_state,
    )

    r_state = rand_state if shuffle is True else None
    cv_out = StratifiedKFold(
        n_splits=k_fold, shuffle=shuffle, random_state=r_state
    )

    with mlflow.start_run(run_name="_".join([model, "_k_fold"])):
        mlflow.log_param("nested_cv", nested_cv)
        if nested_cv is True:
            cv_results = train_with_nested_cv(
                classifier,
                X,
                y,
                cv_out,
                model,
                k_fold_inn,
                shuffle,
                r_state,
            )
        else:
            cv_results = train_without_nested_cv(
                classifier,
                X,
                y,
                cv_out,
                model,
                n_neighbors,
                leaf_size,
                c_reg,
                max_iter,
                n_estimators,
                max_depth,
                criterion,
                max_features,
                k_fold,
            )

        mlflow.log_param("preprocess", preproc)
        mlflow.log_param("n_features", n_features)
        mlflow.log_param("k_fold", k_fold)
        mlflow.log_param("shuffle", shuffle)

    click.echo(f"Cross validation results: {cv_results}")
    classifier.fit(X, y)
    dump(classifier, save_model_path)
    click.echo(f"Model has been saved to {save_model_path}")
