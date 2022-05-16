import pandas as pd
import numpy as np
import re


def replace_1_with_number(column):
    number = re.search(r"\d+", column.name).group()
    return column.replace(1, int(number))


def prepare(X: pd.DataFrame) -> pd.DataFrame:
    df = X.copy()
    df["Soil_Type"] = (
        df.loc[:, "Soil_Type1":"Soil_Type40"]
        .apply(replace_1_with_number)
        .sum(axis=1)
    )
    df["Mean_Elevation_Vertical_Distance_Hydrology"] = df[
        ["Elevation", "Vertical_Distance_To_Hydrology"]
    ].mean(axis=1)
    df["Mean_Distance_Hydrology_Fire_Points"] = df[
        [
            "Horizontal_Distance_To_Hydrology",
            "Horizontal_Distance_To_Fire_Points",
        ]
    ].mean(axis=1)
    df["Mean_Distance_Hydrology_Roadways"] = df[
        ["Horizontal_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways"]
    ].mean(axis=1)
    df["Mean_Distance_Firepoints_Roadways"] = df[
        [
            "Horizontal_Distance_To_Fire_Points",
            "Horizontal_Distance_To_Roadways",
        ]
    ].mean(axis=1)
    df["sqrt_Horizontal_Distance_To_Hydrology"] = np.sqrt(
        df["Horizontal_Distance_To_Hydrology"]
    )
    df["sqrt_Hillshade_3pm"] = np.sqrt(df["Hillshade_3pm"])
    df["sqrt_Slope"] = np.sqrt(df["Slope"])
    df["Log_Elevation"] = np.log(df["Elevation"])
    df["Log_Aspect"] = np.log(df["Aspect"].replace(0, 360))
    # X["Aspect_Cosine"] = X["Aspect"].replace(0, 360).apply(math.cos)
    remove_base_features = [
        "Id",
        "Horizontal_Distance_To_Hydrology",
        "Elevation",
        "Aspect",
        "Hillshade_3pm",
        "Hillshade_9am",  # strongly correlated with Hillshade_3pm
        "Slope",
    ]
    df = df.drop(df.loc[:, "Soil_Type1":"Soil_Type40"], axis=1)
    df = df.drop(columns=remove_base_features)
    return df
