import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings

from keras.preprocessing.sequence import TimeseriesGenerator
from keras import Sequential
from keras.callbacks import History, EarlyStopping
from sklearn.preprocessing import MinMaxScaler, Normalizer, normalize, StandardScaler
from typing import List, Tuple

from model import get_lstm_model

np.set_printoptions(edgeitems=50, linewidth=10000)


class Agent:

    def __init__(self, min_scale: int = 0, max_scale: int = 1, resolution: int = 15,
                 n_prev: int = 96, start_index: int = 96, batch_size: int = 64,
                 target: str = "y", verbose: bool = False,
                 model: Sequential = None, filepath: str = './models/LSTM_model') -> None:
        if start_index < n_prev-1:
            raise ValueError(
                f'Start index ({start_index}) must be greater than or equal to n_prev-1 ({n_prev-1})')
        if resolution != 5 and resolution != 15:
            raise ValueError(
                f'Resolution must be either 5 or 15, got {resolution}')
        print(f'Resolution is {resolution} and n_prev is {n_prev}. This is equivalent to looking back {(n_prev * resolution) // 60} hours and {(n_prev * resolution) % 60} minutes. ')
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.resolution = resolution
        self.n_prev = n_prev
        self.start_index = start_index
        self.batch_size = batch_size
        self.target = target
        self.verbose = verbose
        self.model = model if model is not None else get_lstm_model()
        self.filepath = filepath
        # If resolution is 15 minutes, then we need 8*15 = 120 minutes = 2 hrs.
        # Else we need 24 predictions (24*5=120) to fill the 2 hours
        self.pred_timesteps = 8 if resolution == 15 else 24
        self.scalers = None

    def add_previous_y_to_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add column to x, called previous_y, which is the last y value
        df = df.assign(previous_y=df[self.target].shift(1)).fillna(0)
        return df
    
    def fit_scalers(self, df: pd.DataFrame) -> None:
        scalers = {}
        for i in range(len(df.columns)):
            scaler = MinMaxScaler(feature_range=(self.min_scale, self.max_scale))
            scaler.fit(df[[df.columns[i]]])
            scalers[df.columns[i]] = scaler
        self.scalers = scalers
    
    def transform(self, df: pd.DataFrame, columns: List[str]=None) -> pd.DataFrame:
        if self.scalers is None:
            raise ValueError(
                f'Scalers have not been fit')
        if columns is None:
            columns = df.columns
        for column in columns:
            if column not in df:
                raise ValueError(
                    f'Column {column} not in dataframe')
            if column not in self.scalers:
                raise ValueError(f'Column {column} has not been fitted with a scaler')
            scaler = self.scalers[column]
            df[column] = scaler.transform(df[[column]])
        return df
    
    # def fit_scaler(self, df: pd.DataFrame) -> None:
    #     self.scaler.fit(df)

    # def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
    #     result_df = pd.DataFrame(self.scaler.transform(df), columns=df.columns)
    #     return result_df
    
    # def inverse_transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
    #     return pd.DataFrame(self.scaler.inverse_transform(df), columns=df.columns)

    def make_training_dataset(self, df: pd.DataFrame) -> tf.data.Dataset:
        if self.verbose:
            print(f'\n===== CALLING MAKE_TRAINING_DATASET =====')
        x, y = df.drop(columns=self.target), df[[self.target]]
        if self.verbose:
            print(f"\n\nx in make_training_dataset:\n{x.head()}")
            print(f"y in make_training_dataset:\n{y.head()}\n\n")
        # Extract all rows apart from the very last one (because we don't have its "correct" value)
        features = np.array(x, dtype=np.float32)[:-1]
        labels = np.array(y, dtype=np.float32)
        # Roll the labels array s.t. they (timewise) correspond to the correct inputs
        labels = np.roll(labels, shift=-self.n_prev+1, axis=0)[:-1]
        print(f"Features shape:\n{features.shape}\nLabels shape:\n{labels.shape}")
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=features,
            targets=labels,
            sequence_length=self.n_prev,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size)
        return ds
    
    def make_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.verbose:
            print(f'\n===== CALLING MAKE_DATASET =====')
        x, y = df.drop(columns=self.target), df[[self.target]]
        x_train = []
        y_train = []
        for i in range(0, len(df)-self.n_prev-1, 1):
            if i % 10000 == 0 and self.verbose:
                print(f"i: {i}")
            inputs = x.loc[i+1:i+self.n_prev].to_numpy()
            target = y.loc[i+self.n_prev].to_numpy()
            x_train.append(inputs)
            y_train.append(target)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print('Finished making dataset.\n')
        return x_train, y_train

    def train(self, df: pd.DataFrame, epochs: int = 10) -> History:
        cb1 = EarlyStopping(
            monitor='loss',
            patience=3,
            restore_best_weights=False
        )
        cb2 = tf.keras.callbacks.ModelCheckpoint(
            self.filepath, verbose=1, save_weights_only=True,
            # Save weights, every epoch.
            save_freq='epoch')
        if self.resolution == 15:
            # Drop 2 out of every 3 rows
            df = df.iloc[::3]
        x_train, y_train = self.make_dataset(df)
        return self.model.fit(x_train, y_train, epochs=epochs, callbacks=[cb1, cb2])

    def predict(self, df: pd.DataFrame, index_to_predict: int) -> float:
        row_start_index = max(0, index_to_predict-self.n_prev+1)
        x = np.expand_dims(
            df.loc[row_start_index:index_to_predict].to_numpy(), axis=0)
        model_pred = self.model(x).numpy()
        return float(model_pred[0][0])

    def predict_n_timesteps(self, df: pd.DataFrame, n_timesteps: int = None) -> pd.Series:
        df = df.copy()
        if self.target in df.columns:
            df = df.drop(self.target, axis=1)
        if n_timesteps is None:
            n_timesteps = self.pred_timesteps
        if self.resolution == 15:
            # Drop 2 out of every 3 rows
            df = df.iloc[::3]
            # TODO: make sure to adjust the index of the prediction when cutting 2/3rds of the dataframe
            # Right now, this produces an error.
        results = []
        if self.verbose:
            print('\n===== CALLING PREDICT_N_TIMESTEPS =====')
        for i in range(self.start_index, self.start_index+n_timesteps):
            result = self.predict(df, i)
            df.loc[i+1, 'previous_y'] = result
            results.append(result)
        return self.inverse_transform_df(df)[self.start_index+1:self.start_index+n_timesteps+1]['previous_y']

    def visualize_results(self, y_true: pd.Series, y_pred: pd.Series, n_timesteps: int = None) -> None:
        if n_timesteps is None:
            n_timesteps = self.pred_timesteps
        # [self.start_index:self.start_index+n_timesteps]
        y_true = y_true.to_numpy()[self.start_index:self.start_index+n_timesteps]
        y_pred = y_pred.to_numpy()
        results = pd.DataFrame(
            {'y_true': y_true.ravel(), 'y_pred': y_pred.ravel()})
        results.plot(figsize=(20, 8))
        plt.show()
        

if __name__ == '__main__':
    df_train = pd.read_csv(
        './Project 3/no1_train.csv').drop("start_time", axis=1)
    df_val = pd.read_csv(
        './Project 3/no1_validation.csv').drop("start_time", axis=1)
    # Move column 'y' in df_train and df_val to the end
    df_train = df_train.reindex(
        columns=df_train.drop('y', axis=1).columns.tolist() + ['y'])
    df_val = df_val.reindex(
        columns=df_val.drop('y', axis=1).columns.tolist() + ['y'])
    print(df_train.head())
    print(df_val.head())
    agent = Agent(
        min_scale=0,
        max_scale=1,
        n_prev=24,
        resolution=5,
        start_index=24,
        batch_size=64,
        target='y',
        verbose=True
    )
    y_true = df_val['y']
    agent.fit_scaler(df_train)
    print(f"\nOriginal df_train:\n{df_train.head(15)}")
    print(f"\nOriginal df_val:\n{df_val.head(15)}\n")
    df_train = agent.transform_df(df_train)
    df_val = agent.transform_df(df_val)
    print(f"\nTransformed df_train:\n{df_train.head(15)}")
    print(f"\nTransformed df_val:\n{df_val.head(15)}\n")
    df_train = agent.add_previous_y_to_df(df_train)
    df_val = agent.add_previous_y_to_df(df_val)
    print(f"\nTrans df_train after prev_y:\n{df_train.head(15)}")
    print(f"Trans df_val after prev_y:\n{df_val.head(15)}\n")
    agent.train(df_train, epochs=1)
    result = agent.predict_n_timesteps(df_val, n_timesteps=5)
    print(f"Resulting pd Series:\n{result}\n")
    agent.visualize_results(y_true=y_true, y_pred=result, n_timesteps=5)
