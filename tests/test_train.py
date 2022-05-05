from click.testing import CliRunner
import click
import pytest
import pandas as pd
from pathlib import Path

from forest_ml.train import train
from forest_ml.predict import predict


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_error_for_invalid_preproc(runner: CliRunner) -> None:
    result = runner.invoke(train, ["--preproc", "tsne"])
    assert result.exit_code == 2
    assert "Invalid value for '--preproc'" in result.output


def test_error_for_invalid_n_features(runner: CliRunner) -> None:
    result = runner.invoke(train, ["--n-features", "100"])
    assert result.exit_code == 1


def test_for_valid_case(runner: CliRunner, tmp_path: Path) -> None:
    samle_path = Path("tests/sample.csv").resolve()
    samle_test_path = Path("tests/sample_test.csv").resolve()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        mdl_path = "mdl.joblib"
        result = runner.invoke(
            train,
            [
                "-d",
                str(samle_path),
                "-s",
                str(mdl_path),
                "--nested-cv",
                "False",
                "-m",
                "knn",
                "--n-neighbors",
                "1",
            ],
        )
        if result.exit_code != 0:
            click.echo(result.output)
        assert result.exit_code == 0
        subm_path = tmp_path / "submissions.csv"
        result = runner.invoke(
            predict,
            [
                "-m",
                str(mdl_path),
                "-t",
                str(samle_test_path),
                "-s",
                str(subm_path),
            ],
        )
        y_predict = pd.read_csv(subm_path)
        assert result.exit_code == 0
        assert (
            y_predict["Cover_Type"].isin(range(1, 8)).sum()
            == y_predict.shape[0]
        )
