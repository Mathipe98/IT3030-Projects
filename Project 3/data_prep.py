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
                 n_prev: int = 96, start_index: int = 96, batch_size: int = 64,
                 target: str = "y", verbose: bool=False) -> None:
        if start_index < n_prev:
            raise ValueError(
                f'Start index ({start_index}) must be greater than or equal to n_prev ({n_prev})')
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
        self.start_index = start_index
        self.batch_size = batch_size
        self.target = target
        self.verbose = verbose
        # If resolution is 15 minutes, then we need 8*15 = 120 minutes = 2 hrs.
        # Else we need 24 predictions (24*5=120) to fill the 2 hours
        self.n_preds = 8 if resolution == 15 else 24
        self.x_scaler, self.y_scaler = self.setup_scalers()

    def setup_scalers(self) -> Tuple[MinMaxScaler, MinMaxScaler]:
        # Create scalers for x and y
        x_scaler = MinMaxScaler(feature_range=(self.min_scale, self.max_scale))
        y_scaler = MinMaxScaler(feature_range=(self.min_scale, self.max_scale))
        return x_scaler, y_scaler
    
    def add_previous_y_to_df(self, df: pd.DataFrame, training: bool=True) -> pd.DataFrame:
        if training:
            # Add column to x, called previous_y, which is the last y value
            df = df.assign(previous_y=df[self.target].shift(1)).fillna(0)
        else:
            df = df.assign(previous_y=0)
        return df

    def fit_scalers_to_df(self, df: pd.DataFrame) -> None:
        if "y" not in df.columns:
            raise ValueError(
                f'Dataframe must have target column when fitting scalers (missing "{self.target}")')
        x, y = df.drop("y", axis=1), df[["y"]]
        self.x_scaler.fit(x)
        self.y_scaler.fit(y)

    def transform_x(self, data: pd.DataFrame = None) -> np.ndarray:
        # Scale x
        if data is None:
            raise ValueError('No data provided to transform_x')
        return self.x_scaler.transform(data)

    def transform_y(self, data: pd.DataFrame = None) -> np.ndarray:
        # Scale y
        if data is None:
            raise ValueError('No data provided to transform_y')
        return self.y_scaler.transform(data)

    def inverse_transform_x(self, data: np.ndarray) -> np.ndarray:
        return self.x_scaler.inverse_transform(data)

    def inverse_transform_y(self, data: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(data)

    def make_training_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        if self.verbose:
            print(f'Making training dataset...')
        x = data.drop(columns=self.target)
        y = data[[self.target]]
        # Scale x and y
        x = self.transform_x(x)
        y = self.transform_y(y)
        if self.verbose:
            print(f"y after transformation in making training dataset: {y}")
        # Extract all rows apart from the very last one (because we don't have its "correct" value)
        features = np.array(x, dtype=np.float32)[:-1]
        labels = np.array(y, dtype=np.float32)
        # Roll the labels array s.t. they (timewise) correspond to the correct inputs
        labels = np.roll(labels, shift=-self.n_prev, axis=0)[:-1]
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=features,
            targets=labels,
            sequence_length=self.n_prev,
            shuffle=False,
            batch_size=self.batch_size)
        return ds

    def make_testing_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        if self.verbose:
            print(f'Making testing dataset to predict {self.n_preds} timesteps')
        idx1 = max(self.start_index - self.n_prev, 0)
        idx2 = min(self.start_index+self.n_preds-1, len(data))
        x = data[idx1:idx2]
        # Scale x
        x = self.transform_x(x)
        features = np.array(x, dtype=np.float32)
        if self.n_prev < self.start_index:
            seq_len = self.n_prev
        else:
            seq_len = self.start_index
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=features,
            targets=None,
            sequence_length=seq_len,
            shuffle=False,
            batch_size=self.batch_size)
        return ds

    def train(self, df: pd.DataFrame, model: Sequential = None, epochs: int = 10) -> History:
        if model is None:
            model = get_lstm_model()
        callbacks = EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=False
        )
        ds = self.make_training_dataset(df)
        return model.fit(ds, epochs=epochs, callbacks=[callbacks])

    def predict_2hrs(self, df: pd.DataFrame, model=None) -> np.ndarray:
        if model is None:
            model = get_lstm_model()
        if 'y' in df.columns:
            df = df.drop('y', axis=1)
        results = []
        ds = self.make_testing_dataset(df)
        # model_input has shape [batch_size, timesteps, features]
        model_input = next(iter(ds))
        # model_output will therefore have the same size
        model_output = model(model_input).numpy()
        # Iterate through the batches
        for i in range(model_output.shape[0]):
            # y_pred has shape [timesteps, features]
            y_pred = model_output[i]
            # Set previous_y in the df using this value
            idx1 = self.start_index - self.n_prev + i
            idx2 = (self.start_index-1) - self.n_prev + len(y_pred) + i
            if self.verbose:
                print(f"DF rows before loc (+ 2 extra just for visuals):\n{df.loc[idx1:idx2+2]}")
                print(f"Results of model:\n{y_pred}")
                # Print indeces
                print(f"i: {i}, idx1: {idx1}, idx2: {idx2}")
            df.loc[idx1+1: idx2+1, 'previous_y'] = y_pred
            if self.verbose:
                print(f"DF rows AFTER loc (+ 2 extra just for visuals):\n{df.loc[idx1:idx2+2]}")
            # Finally, only append the very last predicted result, as this is the final timestep prediction
            results.append(y_pred[-1,0])
        # Turn results into a dataframe with column "y"
        results = pd.DataFrame(data=results, columns=['y'])
        # Return the inverse transformed version
        return self.inverse_transform_y(results)

    def visualize_results(self, y_true: pd.Series, y_pred: np.ndarray) -> None:
        results = pd.DataFrame(
            {'y_true': y_true.values[self.start_index:self.start_index+self.n_preds], 'y_pred': y_pred.ravel()})
        results.plot(figsize=(20, 8))


if __name__ == '__main__':
    df_train = pd.read_csv(
        './Project 3/no1_train.csv').drop("start_time", axis=1)
    df_val = pd.read_csv(
        './Project 3/no1_validation.csv').drop("start_time", axis=1)
    agent = Agent(n_prev=96,
              resolution=5,
              start_index=96,
              batch_size=64,
              target='y'
              )
    model = get_lstm_model()
    df_train = agent.add_previous_y_to_df(df_train, training=True)
    df_val = agent.add_previous_y_to_df(df_val, training=False)
    agent.fit_scalers_to_df(df_train)
    agent.train(df_train, model=model, epochs=1)
    testing_ds = agent.make_testing_dataset(df_val.drop(columns="y"))
    results = agent.predict_2hrs(df_val, model)
    print(results)
    agent.visualize_results(df_val['y'], results, start_index=agent.start_index)