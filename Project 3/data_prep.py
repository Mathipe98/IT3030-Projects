import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

from keras.preprocessing.sequence import TimeseriesGenerator
from keras import Sequential
from keras.callbacks import History, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, Normalizer, normalize, StandardScaler
from typing import Tuple

from model import get_lstm_model

class Agent:

    def __init__(self, min_scale: int = 0, max_scale: int = 1, resolution: int = 15,
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
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.resolution = resolution
        self.n_prev = n_prev
        # If resolution is 15 minutes, then we need 8*15 = 120 minutes = 2 hrs.
        # Else we need 24 predictions (24*5=120) to fill the 2 hours
        self.n_preds = 8 if resolution == 15 else 24
        self.batch_size = batch_size
        self.x_scaler, self.y_scaler = self.setup_scalers()

    def setup_scalers(self) -> Tuple[MinMaxScaler, MinMaxScaler]:
        # Create scalers for x and y
        x_scaler = MinMaxScaler(
            feature_range=(self.min_scale, self.max_scale))
        y_scaler = MinMaxScaler(
            feature_range=(self.min_scale, self.max_scale))
        x_scaler = StandardScaler()
        y_scaler = StandardScaler()
        return x_scaler, y_scaler
    
    def fit_scalers_to_df(self, df: pd.DataFrame) -> None:
        if "y" not in df.columns:
            raise ValueError('Dataframe must have a column named "y" when fitting scalers')
        x, y = df.drop("y", axis=1), df[["y"]]
        self.x_scaler.fit(x)
        self.y_scaler.fit(y)

    def transform_x(self, data: pd.DataFrame = None) -> np.ndarray:
        # Scale x
        if data is None:
            raise ValueError('No data provided to transform_x')
        try:
            return self.x_scaler.transform(data)
        except:
            raise RuntimeError('transform_x was called before scalers were fit to data')

    def transform_y(self, data: pd.DataFrame = None) -> np.ndarray:
        # Scale y
        if data is None:
            raise ValueError('No data provided to transform_y')
        try:
            return self.y_scaler.transform(data)
        except:
            raise RuntimeError('transform_y was called before scalers were fit to data')

    def inverse_transform_x(self, data: np.ndarray) -> np.ndarray:
        return self.x_scaler.inverse_transform(data)

    def inverse_transform_y(self, data: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(data)

    def process_df_to_xy(self, df: pd.DataFrame, target: str, training: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        if df is None:
            raise ValueError('Dataframe not provided in process_df_to_xy')
        if training:
            if self.resolution == 15:
                # Remove 2 rows for every 3rd row if we use every 15 minutes
                df = df.copy().iloc[::3]
            x, y = df.drop(target, axis=1), df[[target]]
            # print(f"y in processing function: {y}")
            # If processing training data, then transform (scale) and add previous_y
            x = self.transform_x(x)
            print(f"x after transformation: {x}")
            y = self.transform_y(y).reshape(-1)
            print(f"y after transformation: {y}")
            # Manually shift the series forward by 1
            previous_y = np.insert(y, 0, 0)
            previous_y = np.delete(previous_y, -1)
            x = np.insert(x, -1, previous_y, axis=1)
        else:
            x = df
            x = self.transform_x(x)
            # y = dummy data for the generator (has no functional purpose)
            y = np.zeros(len(x))
        return x, y

    def get_generator(self, df: pd.DataFrame, target: str, training: bool = True) -> TimeseriesGenerator:
        x, y = self.process_df_to_xy(df, target, training)
        print(f"x generated during get_generator: {x}")
        print(f"y generated during get_generator: {y}")
        bsize = self.batch_size
        print(f'Using batch size {bsize}\n')
        return TimeseriesGenerator(x, y, length=self.n_prev, batch_size=bsize)

    def train(self, df: pd.DataFrame, target: str="y", model: Sequential=None, epochs: int=10) -> History:
        if model is None:
            model = get_lstm_model()
        gen = self.get_generator(df, target, training=True)
        callbacks = EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=True
        )
        return model.fit(gen, epochs=epochs, callbacks=[callbacks])

    def predict_2hrs(self, df: pd.DataFrame, model=None, start_index: int=1) -> np.ndarray:
        if start_index < 1:
            raise ValueError('start_index must be >= 1')
        if model is None:
            model = get_lstm_model()
        # Add a column to the test-dataframe with the predicted values
        df['previous_y'] = np.random.uniform(low=0, high=1, size=len(df))
        n_prev = self.n_prev
        # Set the previous y value at start index to the model prediction
        results = []
        for i in range(start_index, min(len(df), start_index+self.n_preds)):
            # Note: use max/min to calculate the correct indices at which to slice the dataframe as well as reshape
            #i-n_prev+1
            idx1 = max(i - n_prev, 0)
            idx2 = min(i, len(df))
            df_slice = df[idx1: idx2].to_numpy()
            # Get size of df_slice
            df_slice = df_slice.reshape(1, min(n_prev, i), len(df.columns))
            result = model(df_slice).numpy().astype(float)[0][0]
            print(f"Result from prediction {i}: {result}\nResult datatype: {type(result)}\n")
            df.loc[i+1, 'previous_y'] = result
            results.append(result)
        # Turn results into a dataframe with column "y"
        results = pd.DataFrame(data=results, columns=['y'])
        return self.transform_y(results)
    
    def visualize_results(self, y_true: pd.DataFrame, y_pred: np.ndarray, start_index: int=1) -> None:
        results = pd.DataFrame({'y_true':y_true.values[start_index:start_index+self.n_preds],'y_pred':y_pred.ravel()})
        results.plot(figsize=(20,8))
