from click.testing import CliRunner
import pytest

from forest_ml.train import train
import forest_ml.eda

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
    # assert "value" in result.output