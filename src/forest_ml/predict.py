import click
from pathlib import Path
from joblib import load
import pandas as pd
import forest_ml.features_preparing as fp


@click.command()
@click.option(
    "-m",
    "--model-path",
    default="data/model.joblib",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File path to saved model .joblib file",
)
@click.option(
    "-t",
    "--test-data-path",
    default="data/test.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="File path to test data .csv file",
)
@click.option(
    "-s",
    "--submission-path",
    default="data/submission.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="File path to save submissions .csv file",
)
def predict(
    model_path: Path, test_data_path: Path, submission_path: Path
) -> None:
# TODO: test predict proba
    model = load(model_path)
    X_test_in = pd.read_csv(test_data_path)
    X_test = fp.prepare(X_test_in)
    click.echo(f"Number of features: {X_test.columns}")
    y_predicted = model.predict(X_test)
    pd.DataFrame({"Id": X_test_in["Id"], "Cover_Type": y_predicted}).set_index(
        ["Id"]
    ).to_csv(submission_path)
