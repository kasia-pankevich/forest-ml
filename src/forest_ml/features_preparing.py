import pandas as pd


def prepare(X: pd.DataFrame) -> pd.DataFrame:
    return X.drop(columns=["Id"])
