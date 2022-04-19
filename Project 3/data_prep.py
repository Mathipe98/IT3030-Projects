from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

class DatasetManager:

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 min_scale: int = 0, max_scale: int = 1, resolution: int = 15,
                 n_prev: int = 96,
                 batch_size: int = 64) -> None:
        if resolution != 5 and resolution != 15:
            raise ValueError(
                f'Resolution must be either 5 or 15, got {resolution}')
        if resolution == 5 and n_prev != 288:
            # Raise warning
            warnings.warn(
                f'Resolution is 5 and n_prev is {n_prev}. This is equivalent to looking back {(n_prev * resolution) // 60} hours and {(n_prev * resolution) % 60} minutes. '
                'If you want to use the previous 24hrs, set n_prev to 288.')
        if resolution == 15 and n_prev != 96:
            # Raise warning
            warnings.warn(
                f'Resolution is 15 and n_prev is {n_prev}. This is equivalent to looking back {(n_prev * resolution) // 60} hours and {(n_prev * resolution) % 60} minutes. '
                'If you want to use the previous 24hrs, set n_prev to 96.')
        self.df_train = df_train
        self.df_test = df_test
        self.resolution = resolution
        if resolution == 15:
            # Remove 2 rows for every 3rd row if we use every 15 minutes
            self.df_train = self.df_train.iloc[::3]
            self.df_test = self.df_test.iloc[::3]
        # Create self.x_train, y_train, x_test, y_test by extracting "y" column from dataframes
        self.x_train = df_train.drop(columns=['y'])
        self.y_train = df_train[['y']]
        self.x_test = df_test.drop(columns=['y'])
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.n_prev = n_prev
        self.batch_size = batch_size
        self.x_scaler, self.y_scaler = self.setup_scalers()

    def setup_scalers(self) -> Tuple[MinMaxScaler, MinMaxScaler]:
        # Create scalers for x and y
        x_scaler = MinMaxScaler(
            feature_range=(self.min_scale, self.max_scale))
        y_scaler = MinMaxScaler(
            feature_range=(self.min_scale, self.max_scale))
        x_scaler.fit(self.x_train)
        y_scaler.fit(self.y_train)
        return x_scaler, y_scaler

    def transform_x(self, data: pd.DataFrame = None) -> np.ndarray:
        # Scale x
        if data is None:
            data = self.x_train
        return self.x_scaler.transform(data)

    def transform_y(self, data: pd.DataFrame = None) -> np.ndarray:
        # Scale y
        if data is None:
            data = self.y_train
        return self.y_scaler.transform(data)

    def inverse_transform_x(self, data: np.ndarray) -> np.ndarray:
        return self.x_scaler.inverse_transform(data)

    def inverse_transform_y(self, data: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(data)

    def process_dataset(self, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if training:
            # If processing training data, then transform (scale) and add previous_y
            x = self.transform_x(self.x_train)
            y = self.transform_y(self.y_train).reshape(-1)
            previous_y = np.insert(y, 0, 0)
            previous_y = np.delete(previous_y, -1)
            x = np.insert(x, -1, previous_y, axis=1)
        else:
            # If testing data, then only scale
            x = self.transform_x(self.x_test)
            y = np.zeros(len(x))
        return x, y

    def get_generator(self, training: bool = True) -> TimeseriesGenerator:
        x, y = self.process_dataset(training)
        bsize = self.batch_size
        print(f'Using batch size {bsize}')
        return TimeseriesGenerator(x, y, length=self.n_prev, batch_size=bsize)


if __name__ == "__main__":
    # Open no1_train.csv and no1_validation.csv
    df_train = pd.read_csv('./Project 3/no1_train.csv')
    df_test = pd.read_csv('./Project 3/no1_validation.csv')
    # Drop start time from both
    df_train.drop(df_train.columns[0], axis=1, inplace=True)
    df_test.drop(df_test.columns[0], axis=1, inplace=True)
    manager = DatasetManager(df_train, df_test, resolution=15, n_prev=96, batch_size=64)
    generator = manager.get_generator(training=False)
    # Print generator shape
    print(generator[0][0].shape)
