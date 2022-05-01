import click
import pandas as pd
from pandas_profiling import ProfileReport

@click.command()
@click.option("-p", "--ds-path", default="data/train.csv")
@click.option("-f", "--output-filename", default="forest_report.html")
def profile(ds_path: str, output_filename: str) -> None:
    data = pd.read_csv(ds_path)
    profile = ProfileReport(data, title="Forest Cover Type Profiling Report", explorative=True)
    profile.to_file(output_filename)
    click.echo(f"View report in file '{output_filename}'")
