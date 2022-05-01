import click
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

@click.command()
@click.option("-p", "--ds-path", default="data/train.csv")
@click.option("-n", "--n-neighbors", default=5)
@click.option("-r", "--rand-state", default=42)
def train(ds_path: str, n_neighbors: int, rand_state) -> None:
    data = pd.read_csv(ds_path)
    click.echo(f"Dataset shape: {data.shape}")

    X, y = data.drop(columns=["Id", "Cover_Type"]), data["Cover_Type"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)
    knn = KNeighborsClassifier(n_neighbors)
    knn.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, knn.predict(X_test))
    click.echo(f"Accuracy: {accuracy}")