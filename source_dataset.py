import pandas as pd
from pathlib import Path

import argparse

import datetime
from dateutil.relativedelta import relativedelta
import requests

import mlflow


def source_dataset(date_list: list[str], base_url: str):
    # full_df = pd.concat(
    #    pd.read_parquet(parquet_file)
    #    for parquet_file in list_of_parquet_files
    # )

    write_path = "data"
    extension = ".parquet"

    with mlflow.start_run() as mlrun:
        mlflow.log_param("data_months", date_list)
        for f in date_list:
            artifact_path = Path(write_path, f"{f}{extension}")
            r = requests.get(f"{base_url}{f}{extension}", allow_redirects=True)
            with open(artifact_path, "wb") as f:
                f.write(r.content)
            mlflow.log_artifact(artifact_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_date",
        "-sd",
        type=str,
        help=f"Start date of the dataset to consider. Should be in format: YYYY-MM",
    )
    parser.add_argument(
        "--number_of_months",
        "-nm",
        type=int,
        default=1,
        help=f"Number of months to use in the dataset. Should be a Natural Number.",
    )

    args = parser.parse_args()

    start_date = args.start_date
    number_of_months = args.number_of_months

    assert number_of_months > 0, "Number of months must be positive"

    if start_date is None:
        start_date_datetime = datetime.date.today()
    else:
        start_date_datetime = datetime.datetime.strptime(start_date, "%Y-%m")

    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_"

    date_list = [
        (start_date_datetime + relativedelta(months=inc)).strftime("%Y-%m")
        for inc in range(number_of_months)
    ]

    source_dataset(date_list, base_url)
