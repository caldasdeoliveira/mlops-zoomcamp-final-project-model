from unicodedata import category
import pandas as pd

from pathlib import Path


def calculate_trip_duration(df: pd.DataFrame):
    df["duration"] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df = df.drop(columns=["tpep_dropoff_datetime"])

    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    return df


def read_dataframe(filename: str):  # TODO correctly type path
    df = pd.read_parquet(filename)

    # manually defining types
    df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
    df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    # calculate target
    df = calculate_trip_duration(df)

    # outlier removal
    df = df[(df.duration >= 1) & (df.duration <= 60)]

    return df


def X_y_split(df: pd.DataFrame, target: str = "duration"):
    assert (
            target in list(df.columns)
        ), f"Target columns provided not in dataframe: {target}"

    X = df.drop(columns=target)
    y = df[target]
    return X, y

if __name__=="__main__":
    