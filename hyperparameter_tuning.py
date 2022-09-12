import pandas as pd
import numpy as np

from pathlib import Path

import mlflow

from model import instantiate_pipeline

from load_data import read_dataframe, X_y_split

# TODO expand and use proper gridsearch and time series split

data_dir = Path("data")
dataset_file_list = [str(x) for x in list(data_dir.glob("*.parquet"))]
dataset_file_list.sort()

results = [0] * len(dataset_file_list[:-1])

for n, f in enumerate(dataset_file_list[:-1]):
    df = read_dataframe(f)

    X, y = X_y_split(df)

    clf = instantiate_pipeline(X)

    clf.fit(X, y)

    df_test = read_dataframe(dataset_file_list[n + 1])

    X_test, y_test = X_y_split(df_test)

    result = clf.score(X_test, y_test)
