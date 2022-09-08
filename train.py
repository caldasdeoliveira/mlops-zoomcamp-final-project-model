import pandas as pd
import numpy as np

from pathlib import Path

import mlflow

from model import instantiate_pipeline

from load_data import read_dataframe, X_y_split

# see available datasets
data_dir = Path("data")
dataset_file_list = [str(x) for x in list(data_dir.glob("*.parquet"))]
dataset_file_list.sort()

print(dataset_file_list)

# training
with mlflow.start_run() as run:
    mlflow.sklearn.autolog()
    df = read_dataframe(dataset_file_list[-1])
    
    df = df[['VendorID', 'tpep_pickup_datetime',
             'trip_distance', 'PULocationID', 'DOLocationID', 'duration',]]
    #TODO move this to a better location
    
    X, y = X_y_split(df)
    
    clf = instantiate_pipeline(X)

    clf.fit(X, y)
