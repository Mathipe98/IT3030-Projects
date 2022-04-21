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
                 n_prev: int = 96, start_index: int = 1, batch_size: int = 64,
                 target: str = "y") -> None:
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
        # If resolution is 15 minutes, then we need 8*15 = 120 minutes = 2 hrs.
        # Else we need 24 predictions (24*5=120) to fill the 2 hours
        self.n_preds = 8 if resolution == 15 else 24
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
            raise ValueError(
                'Dataframe must have a column named "y" when fitting scalers')
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
            raise RuntimeError(
                'transform_x was called before scalers were fit to data')

    def transform_y(self, data: pd.DataFrame = None) -> np.ndarray:
        # Scale y
        if data is None:
            raise ValueError('No data provided to transform_y')
        try:
            return self.y_scaler.transform(data)
        except:
            raise RuntimeError(
                'transform_y was called before scalers were fit to data')

    def inverse_transform_x(self, data: np.ndarray) -> np.ndarray:
        return self.x_scaler.inverse_transform(data)

    def inverse_transform_y(self, data: np.ndarray) -> np.ndarray:
        return self.y_scaler.inverse_transform(data)

    def make_training_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        print("Generating training dataset, therefore start_index has no effect.")
        x = data.drop(columns=self.target)
        y = data[self.target]
        # Extract all rows apart from the very last one (because we don't have its "correct" value)
        features = np.array(x, dtype=np.float32)[:-1]
        labels = np.array(y, dtype=np.float32)
        # Roll the labels array s.t. they (timewise) correspond to the correct inputs
        labels = np.roll(labels, shift=-self.n_prev, axis=0)[:-1]
        # Note: first n_prev elements of labels go unused
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=features,
            targets=labels,
            sequence_length=self.n_prev,
            shuffle=False,
            batch_size=self.batch_size)
        return ds

    def make_testing_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        idx1 = max(self.start_index - self.n_prev, 0)
        idx2 = min(self.start_index, len(data))
        # Print indeces
        print(f"idx1: {idx1}, idx2: {idx2}")
        x = data.drop(columns=self.target)[idx1:idx2]
        print(f"Resulting x: {x}")
        features = np.array(x, dtype=np.float32)
        print(f"Actual target with these indices: {data[self.target][idx2]}")
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
        # Add a column to the test-dataframe with the predicted values
        # They will be initialized randomly since we update them as we go
        df['previous_y'] = np.random.uniform(low=0, high=1, size=len(df))
        n_prev = self.n_prev
        # Set the previous y value at start index to the model prediction
        results = []
        for i in range(self.start_index, min(len(df), self.start_index+self.n_preds)):
            idx1 = max(i - n_prev, 0)
            idx2 = min(i, len(df))
            df_slice = df[idx1: idx2].to_numpy()
            # Get size of df_slice
            df_slice = df_slice.reshape(1, min(n_prev, i), len(df.columns))
            # We will get a 3D output with batch size 1 -> resize to 2D
            result = model(df_slice).numpy().reshape(df_slice.shape[1], -1)
            # We now have shape [timesteps, predictions]. Set this as previous_y
            df.loc[idx1+1: idx2, 'previous_y'] = result
            # Finally, only append the very last predicted result, as this is the final timestep prediction
            results.append(result[-1,0])
        # Turn results into a dataframe with column "y"
        results = pd.DataFrame(data=results, columns=['y'])
        # TODO: actually return scaled version
        return results

    def visualize_results(self, y_true: pd.DataFrame, y_pred: np.ndarray, start_index: int = 1) -> None:
        results = pd.DataFrame(
            {'y_true': y_true.values[start_index:start_index+self.n_preds], 'y_pred': y_pred.ravel()})
        results.plot(figsize=(20, 8))


if __name__ == '__main__':
    df_train = pd.read_csv(
        './Project 3/no1_train.csv').drop("start_time", axis=1)
    df_val = pd.read_csv(
        './Project 3/no1_validation.csv').drop("start_time", axis=1)
    agent = Agent(n_prev=10,
              resolution=5,
              start_index=10,
              batch_size=64,
              target='y'
              )
    print(df_val.head(20))
    model = get_lstm_model()
    results = agent.predict_2hrs(df_val, model)
    print(results)
    # agent.train(df_train, model, epochs=5)
