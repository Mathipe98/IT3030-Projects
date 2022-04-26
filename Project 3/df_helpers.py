from typing import List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.interpolate import splev, splrep


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


def spline_interpolate(df: pd.DataFrame, features: List[str], market_time: str='hourly') -> pd.DataFrame:
    if market_time != 'hourly' and market_time != 'quarterly':
        raise ValueError('market_time must be either hourly or quarterly')
    # Make a copy of the dataframe with the only feature we care about
    df = df.copy()[features]
    original_df = df.copy()
    df['sum'] = df.sum(axis=1)
    if market_time == 'hourly':
        first_sum = df['sum'][0]
        df = df[6:]
        df = df.iloc[::12]
    if market_time == 'quarterly':
        df = df.iloc[::3]
    # Add a new column which is the sum of all other columns
    x = np.insert(np.array(df.index), 0, 0, axis=0)
    y = np.insert(np.array(df['sum']), 0, first_sum, axis=0)
    spl = splrep(x, y, k=5)
    x2 = np.linspace(original_df.index[0], original_df.index[-1], len(original_df))
    y2 = splev(x2, spl)
    result = pd.DataFrame(y, index=x, columns=['sum'])
    result = pd.concat([original_df, result], ignore_index=False, axis=1).fillna(method='ffill')
    result['interpolated'] = y2
    result = result[['interpolated', 'sum']]
    return result

def add_altered_forecast(df: pd.DataFrame, features: List[str], market_time: str='hourly') -> pd.DataFrame:
    inter_df = spline_interpolate(df, features, market_time)
    inter_df['diff'] = inter_df['interpolated'] - inter_df['sum']
    df['altered_imbalance'] = df['y'] - inter_df['diff']
    # Shift by 1
    df['previous_altered_imbalance'] = df['altered_imbalance'].shift(1).fillna(0)
    return df