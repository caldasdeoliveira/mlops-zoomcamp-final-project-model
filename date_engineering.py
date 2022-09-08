import pandas as pd

class DateEngineering():
    def __init__(self, date_col: list[str] = "tpep_pickup_datetime") -> None:
        self.date_col = date_col

    def fit(self, X=None, y=None):
        pass

    def transform(self, X: pd.DataFrame):
        assert (
            self.date_col in list(X.columns)
        ), f"Datetime columns provided to date_engineering not in dataframe: {set(self.date_col)-set(X.columns)}"

        X['Year'] = X[self.date_col].dt.year

        X['Month'] = X[self.date_col].dt.month

        X['Day'] = X[self.date_col].dt.day

        X['Day_of_week'] = X[self.date_col].dt.dayofweek

        X['Day_of_year'] = X[self.date_col].dt.dayofyear

        X['Hour'] = X[self.date_col].dt.hour

        X['Minute'] = X[self.date_col].dt.minute

        X = X.drop(columns=self.date_col)

        return X
    
    def fit_transform(self, X: pd.DataFrame, y=None):
        return self.transform(X)