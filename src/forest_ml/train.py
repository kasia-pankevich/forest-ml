import click
from pathlib import Path
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from joblib import dump

@click.command()
@click.option("-p", "--ds-path", default="data/train.csv", 
                type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-n", "--n-neighbors", default=5, type=int)
@click.option("-r", "--rand-state", default=42, type=int)
@click.option("-s", "--save-model-path", default="data/model.joblib",
                type=click.Path(dir_okay=False, writable=True, path_type=Path))
def train(ds_path: Path, n_neighbors: int, rand_state: int, save_model_path: Path) -> None:
    data = pd.read_csv(ds_path)
    click.echo(f"Dataset shape: {data.shape}")

    X, y = data.drop(columns=["Id", "Cover_Type"]), data["Cover_Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, knn.predict(X_test))
    click.echo(f"Accuracy: {accuracy}")
    dump(knn, save_model_path)