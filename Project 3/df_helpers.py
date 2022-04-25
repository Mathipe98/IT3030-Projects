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

def add_shifted_target(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df[f'previous_{target}'] = df[target].shift(1).fillna(0)
    return df

def add_shift(df: pd.DataFrame, feature: str, hours: int) -> pd.DataFrame:
    """
    Adds a column to the dataframe that is the shifted value of the given feature.
    """
    df[f'{feature}_shift_{hours}'] = df[feature].shift((60*hours)//5).fillna(0)
    return df

def add_rolling_avg(df: pd.DataFrame, feature: str, hours: int) -> pd.DataFrame:
    """
    Adds a column to the dataframe that is the average of the previous y values.
    """
    df[f'{feature}_avg_prev_{hours}h'] = df[feature].rolling((60*hours)//5).mean().fillna(0)
    return df