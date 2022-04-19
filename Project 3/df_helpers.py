import pandas as pd


def get_outliers(df: pd.DataFrame, col: str, threshold: float = 3) -> pd.DataFrame:
    """
    Returns a dataframe of rows that are outliers in the given column.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - (threshold * iqr)
    upper_bound = q3 + (threshold * iqr)
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]

def get_rows_between(df: pd.DataFrame, feature: str, lower: float, upper: float) -> pd.DataFrame:
    """
    Returns a dataframe of rows that are between the given lower and upper bounds.
    """
    return df[(df[feature] >= lower) & (df[feature] <= upper)]