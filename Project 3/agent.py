import df_helpers as dfh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Sequential
from keras.callbacks import History, EarlyStopping
from multiprocessing.pool import ThreadPool as Pool
from sklearn.preprocessing import MinMaxScaler
from typing import List, Tuple


np.set_printoptions(edgeitems=50, linewidth=10000)
np.random.seed(314159)


class Agent:

    def __init__(self, min_scale: int = 0, max_scale: int = 1, resolution: int = 15,
                 n_prev: int = 96, batch_size: int = 64,
                 target: str = "y", verbose: bool = False,
                 model: Sequential=None, filepath: str = './models/LSTM_model') -> None:
        if resolution != 5 and resolution != 15:
            raise ValueError(
                f'Resolution must be either 5 or 15, got {resolution}')
        if model is None:
            raise ValueError(
                f'Model must be specified, got {model}')
        print(f'Resolution is {resolution} and n_prev is {n_prev}. This is equivalent to looking back {(n_prev * resolution) // 60} hours and {(n_prev * resolution) % 60} minutes. ')
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.resolution = resolution
        self.n_prev = n_prev
        self.batch_size = batch_size
        self.target = target
        self.verbose = verbose
        self.model = model
        self.filepath = filepath
        # If resolution is 15 minutes, then we need 8*15 = 120 minutes = 2 hrs.
        # Else we need 24 predictions (24*5=120) to fill the 2 hours
        self.pred_timesteps = 8 if resolution == 15 else 24
        self.scalers = None
    
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
    
    def generate_splits(self, index: int):
            if index % 10000 == 0 and self.verbose:
                print(f'Processing chunk {index}')
            inputs = self.x.loc[index+1:index+self.n_prev].to_numpy()
            target = self.y.loc[index+self.n_prev].to_numpy()
            return inputs, target
    
    def make_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.verbose:
            print(f'\n===== CALLING MAKE_DATASET =====')
        x, y = df.drop(columns=self.target), df[[self.target]]
        self.x = x
        self.y = y
        indeces = [i for i in range(0, len(df)-self.n_prev-1)]
        pool = Pool(processes=10)
        result = pool.map(self.generate_splits, indeces)
        x_train = np.array([i[0] for i in result])
        y_train = np.array([i[1] for i in result])
        print('Finished making dataset.\n')
        return x_train, y_train

    def train(self, train: pd.DataFrame, valid: pd.DataFrame, epochs: int = 10, force_relearn: bool=False) -> History:
        try:
            if force_relearn:
                # Forcefully execute except-clause
                print('Forcing relearn...')
                raise Exception()
            self.model.load_weights(self.filepath)
            if self.verbose:
                print(f'\n===== LOADED MODEL FROM {self.filepath} =====')
            return self.model.history
        except:
            if self.verbose:
                print(f"Could not load model from {self.filepath}. Training new model.")
            cb1 = EarlyStopping(
                monitor='val_loss',
                patience=30,
                restore_best_weights=False
            )
            # cb2 = tf.keras.callbacks.ModelCheckpoint(self.filepath, verbose=1)
            if self.resolution == 15:
                # Drop 2 out of every 3 rows
                train = train.iloc[::3]
            if self.verbose:
                print('Generating training data...')
            x_train, y_train = self.make_dataset(train)
            if self.verbose:
                print('Generating validation data...')
            x_valid, y_valid = self.make_dataset(valid)
            history = self.model.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=epochs, callbacks=[cb1])
            self.model.save_weights(self.filepath)
            return history

    def predict(self, df: pd.DataFrame, index_to_predict: int) -> float:
        row_start_index = max(0, index_to_predict-self.n_prev+1)
        x = np.expand_dims(
            df.loc[row_start_index:index_to_predict].to_numpy(), axis=0)
        model_pred = self.model(x).numpy()
        return float(model_pred[0][0])

    def predict_n_timesteps(self, df: pd.DataFrame, n_timesteps: int = None, start_index: int=None, replace: bool=True) -> np.array:
        if start_index is None:
            start_index = 500
        if start_index < self.n_prev-1:
            raise ValueError(
                f'Start index ({start_index}) must be greater than or equal to n_prev-1 ({self.n_prev-1})')
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
        for i in range(start_index, start_index+n_timesteps):
            result = self.predict(df, i)
            if replace:
                df.loc[i+1, 'previous_y'] = result
                # Since we added features based on previous_y values, we must update them when replacing
                df = dfh.add_rolling_avg(df, feature='previous_y', hours=24)
                df = dfh.add_rolling_avg(df, feature='previous_y', hours=12)
                df = dfh.add_rolling_avg(df, feature='previous_y', hours=6)
                df = dfh.update_altered_forecast(df, features=['total', 'flow'])
            results.append(result)
        target_scaler = self.scalers[self.target]
        # Undo the scaling
        results = target_scaler.inverse_transform(np.array(results).reshape(-1, 1))
        return results

    def visualize_results(self, y_true: np.array, y_pred: np.array, n_timesteps: int = None, start_index: int=None, ax=None) -> None:
        if n_timesteps is None:
            n_timesteps = self.pred_timesteps
        if start_index is None:
            start_index = 500
        # [self.start_index:self.start_index+n_timesteps]
        y_hist = y_true[start_index-self.n_prev:start_index].ravel()
        y_true = y_true[start_index:start_index+n_timesteps].ravel()
        y_pred = y_pred.ravel()
        # Create helper lists and add np.nan before and after to properly set plot starting points
        hist_list = [n for n in y_hist]
        hist_list.append(y_true[0])
        nan_list = []
        for i in range(self.n_prev):
            if i != self.n_prev-1:
                hist_list.append(np.nan)
            nan_list.append(np.nan)
        nan_list = np.array(nan_list)
        y_hist = np.array(hist_list)
        y_true = np.concatenate((nan_list, y_true))
        y_pred = np.concatenate((nan_list, y_pred))
        results = pd.DataFrame(
            {'y_hist': y_hist, 'y_true': y_true, 'y_pred': y_pred})
        if ax is None:
            results.plot(figsize=(12, 6))
            plt.show()
        else:
            ax.set_title(f'Predict with start index: {start_index}')
            results.plot(ax=ax)
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Imbalance')
    
    def visualize_multiple_predictions(self, df: pd.DataFrame, y_true: np.ndarray, n_timesteps: int=None, replace: bool=True, n_batches: int=10,) -> None:
        # Make a list with 10 different random numbers between self.n_prev and len(df)-self.n_prev
        random_indeces = np.random.randint(
            self.n_prev, len(df)-self.n_prev, n_batches)
        _, axes = plt.subplots(nrows=2, ncols=n_batches//2, figsize=(20, 8))
        axes = axes.ravel()
        preds = []
        for index in random_indeces:
            preds.append(self.predict_n_timesteps(df, n_timesteps=n_timesteps, start_index=index, replace=replace))
        for i in range(len(preds)):
            self.visualize_results(y_true, y_pred=preds[i], n_timesteps=n_timesteps, start_index=random_indeces[i], ax=axes[i])
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
    agent = Agent(
        min_scale=0,
        max_scale=1,
        n_prev=24,
        resolution=5,
        batch_size=64,
        target='y',
        verbose=True
    )
    # agent.visualize_multiple_predictions(df_train, n_batches=10)
