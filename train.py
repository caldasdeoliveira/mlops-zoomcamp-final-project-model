import pandas as pd
import numpy as np

from pathlib import Path
import argparse

import mlflow

from model import instantiate_pipeline

# see available datasets
# data_dir = Path("data")
# dataset_file_list = [str(x) for x in list(data_dir.glob("*.parquet"))]
# dataset_file_list.sort()

# print(dataset_file_list)

parser = argparse.ArgumentParser()
parser.add_argument("--X_dataset", "-X", type=str)
parser.add_argument("--y_dataset", "-y", type=str)

args = parser.parse_args()

X_path = args.X_dataset
y_path = args.y_dataset

# training
with mlflow.start_run() as run:
    mlflow.sklearn.autolog()
    X = pd.read_parquet(X_path)
    y = pd.read_parquet(y_path)

    clf = instantiate_pipeline(X)

    clf.fit(X, y)
