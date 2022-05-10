import click
import pandas as pd
from pandas_profiling import ProfileReport


@click.command()
@click.option(
    "-d",
    "--ds-path",
    default="data/train.csv",
    help="Path to train dataset csv",
)
@click.option(
    "-s",
    "--save-output-path",
    default="forest_report.html",
    help="Local folder path and file name to save EDA report to",
)
def profile(ds_path: str, save_output_path: str) -> None:
    data = pd.read_csv(ds_path)
    profile = ProfileReport(
        data, title="Forest Cover Type Profiling Report", explorative=True, minimal=False
    )
    profile.to_file(save_output_path)
    click.echo(f"View report in file '{save_output_path}'")
