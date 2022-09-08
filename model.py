import imp
from typing import List
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from date_engineering import DateEngineering

from sklearn.ensemble import GradientBoostingRegressor

def instantiate_pipeline(
    df: pd.DataFrame,
    cat_cols: list[str] = ["PULocationID", "DOLocationID"],
    date_cols: list[str] = ["tpep_pickup_datetime"],
    num_cols: list[str] = None,
):

    assert (
        set(cat_cols).issubset( list(df.columns) )
    ), f"Categorical columns provided to ColumnTransformer not in dataframe: {set(cat_cols)-set(df.columns)}"
    categorical_features = cat_cols
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")
    #TODO use another encoder to avoid the curse of dimensionality

    assert (
        set(date_cols).issubset( list(df.columns) )
    ), f"Datetime columns provided to ColumnTransformer not in dataframe: {set(date_cols)-set(df.columns)}"
    datetime_features = date_cols
    datetime_transformer = Pipeline(steps=[("datetime", DateEngineering())])

    if num_cols is None:
        numeric_features = list(set(df.columns) - set(categorical_features) - set(datetime_features))
    else: 
        numeric_features = num_cols
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("date", datetime_transformer, datetime_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    clf = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingRegressor()),
        ]
    )
    return clf
