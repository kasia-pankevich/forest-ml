import click
import pandas as pd

@click.command()
@click.option("-p", "--ds-path", default="data/train.csv")
def train(ds_path: str) -> None:
    data = pd.read_csv(ds_path)
    click.echo(f"Dataset shape: {data.shape}")