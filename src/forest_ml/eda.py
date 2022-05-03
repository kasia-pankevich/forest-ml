import click
import pandas as pd
from pandas_profiling import ProfileReport

@click.command()
@click.option("-d", "--ds-path", default="data/train.csv",
                help="Path to train dataset csv")
@click.option("-s", "--save-output-path", default="forest_report.html",
                help="Local folder path and file name to save model to")
def profile(ds_path: str, output_filename: str) -> None:
    data = pd.read_csv(ds_path)
    profile = ProfileReport(data, title="Forest Cover Type Profiling Report", explorative=True)
    profile.to_file(output_filename)
    click.echo(f"View report in file '{output_filename}'")
