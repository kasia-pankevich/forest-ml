import pandas as pd
import numpy as np
import math
import re
from sklearn.preprocessing import OneHotEncoder

def add_soil_family_and_rock_type(df: pd.DataFrame) -> pd.DataFrame:
    soil_family_map = {
        "Cathedral": [1],
        "Ratake": [2],
        "Rock_Outcrop": [3, 4, 5, 6, 35, 37],
        "Gothic": [7],
        "Limber": [8],
        "Troutville": [9],
        "Catamount": [10, 11, 13, 26],
        "Legault": [12],
        "Aquolis": [14],
        "Unspecified": [15],
        "Cryaquolis": [16, 17, 19, 20, 21],
        "Rogert": [18],
        "Leighcan": [22, 23, 24, 25, 27, 28, 31, 32, 33, 38, 39],
        "Como": [29, 30],
        "Rock_Land": [34, 36, 40],
    }
    soil_rock_type_map = {
        "Another": [7, 8, 14, 15, 16, 17, 19, 20, 21, 23, 35],
        "Rubby": [3, 4, 5, 10, 11, 13],
        "Stony": [1, 2, 6, 9, 12, 18, 22, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 36, 37, 38, 39, 40]
    }
    soil_family_map_inv = {}
    for k, v_list in soil_family_map.items():
        for v in v_list:
            soil_family_map_inv[v] = "Soil_Family_" + k
            
    soil_rock_type_map_inv = {}
    for k, v_list in soil_rock_type_map.items():
        for v in v_list:
            soil_rock_type_map_inv[v] = "Soil_Rock_Type_" + k
            
    def replace_1_with_number(column):
        number = re.search("\d+", column.name).group()
        return column.replace(1, int(number))
        
    soil_type = df.loc[:, "Soil_Type1":"Soil_Type40"].apply(replace_1_with_number).sum(axis=1)

    soil_family = soil_type.map(soil_family_map_inv).to_numpy().reshape(-1, 1)
    ohe = OneHotEncoder(drop="first").fit(soil_family)
    soil_family_df = pd.DataFrame(ohe.transform(soil_family).toarray(), columns=ohe.categories_[0][1:])

    soil_rock_type = soil_type.map(soil_rock_type_map_inv).to_numpy().reshape(-1, 1)
    ohe = OneHotEncoder(drop="first").fit(soil_rock_type)
    soil_rock_type_df = pd.DataFrame(ohe.transform(soil_rock_type).toarray(), columns=ohe.categories_[0][1:])

    return pd.concat([
        df, 
        # soil_family_df, 
        soil_rock_type_df
    ], axis=1)


def prepare(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["Elevation_Log"] = np.log(X["Elevation"])
    X["Aspect_Group"] = pd.cut(X["Aspect"], bins=[-0.1, 50, 100, 160, 230, 300, 360],
                           labels=[25, 75, 130, 200, 265, 330])
    X["Aspect_Cosine"] = X["Aspect"].apply(math.cos)
    X["Aspect_Cosine_Group"] = X["Aspect_Group"].apply(math.cos)
    X["Hillshade_Mean"] = X[["Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm"]].mean(axis=1)
    X["Euclidean_Distance_To_Hydrology"] = np.linalg.norm(X[["Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"]], axis=1)
    # X["Hillshade_9am_sq"] = X["Hillshade_9am"]**2
    # X["Hillshade_Noon_sq"] = X["Hillshade_Noon"]**2
    # X["Hillshade_3pm_sq"] = X["Hillshade_3pm"]**2
    X = add_soil_family_and_rock_type(X)
    X = X.drop(columns=["Id", "Aspect", "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", 
                        "Elevation", "Horizontal_Distance_To_Hydrology", "Vertical_Distance_To_Hydrology"])
    X = X.drop(X.loc[:, "Soil_Type1":"Soil_Type40"], axis=1)
    return X
    # return X[['Elevation_Log', 'Hillshade_Mean', 'Euclidean_Distance_To_Hydrology', 
    #    'Hillshade_9am_sq', 'Hillshade_Noon_sq', 'Hillshade_3pm_sq', 'Aspect', 'Slope',
    #    'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points',
    #    'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
    #    'Wilderness_Area4']]
