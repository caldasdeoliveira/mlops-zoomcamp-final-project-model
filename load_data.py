from unicodedata import category
import pandas as pd
import argparse

from pathlib import Path
import mlflow


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
    assert target in list(
        df.columns
    ), f"Target columns provided not in dataframe: {target}"

    X = df.drop(columns=target)
    y = df[target]
    return X, y


def filter_columns(df: pd.DataFrame, relevant_columns=None):
    if relevant_columns is None:
        relevant_columns = [
            "VendorID",
            "tpep_pickup_datetime",
            "trip_distance",
            "PULocationID",
            "DOLocationID",
            "duration",
        ]
    else:
        assert set(relevant_columns).issubset(
            df.columns
        ), f"Relevant columns to use not in dataframe: {set(relevant_columns)-set(df.columns)}"
    df = df[relevant_columns]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", "-f", type=str)

    args = parser.parse_args()

    filename = args.filename

    file_path = Path(filename)  # TODO clean up code
    basename = file_path.name

    base_data_path = "data"

    with mlflow.start_run() as mlrun:
        df = read_dataframe(filename)
        df = filter_columns(df)

        X, y = X_y_split(df)
        y = y.to_frame()

        X_parquet = Path(base_data_path, f"X_{basename}")
        X.to_parquet(X_parquet)
        y_parquet = Path(base_data_path, f"y_{basename}")
        y.to_parquet(y_parquet)

        mlflow.log_artifacts(X_parquet)
        mlflow.log_artifacts(y_parquet)
