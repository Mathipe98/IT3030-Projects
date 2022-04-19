from typing import Tuple
import numpy as np
import pandas as pd
import tensorflow as tf

from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler


# class WindowGenerator():
#     def __init__(self, df_train, df_test, resolution: int = 15,
#                  n_prev: int = 96, pred_size: int = 1, pred_offset: int = 1,
#                  label_columns=None):
#         # Store the raw data.
#         self.train_df = df_train
#         # self.val_df = df_val
#         self.test_df = df_test

#         # Work out the label column indices.
#         self.label_columns = label_columns
#         # if label_columns is not None:
#         #     self.label_columns_indices = {name: i for i, name in
#         #                                   enumerate(label_columns)}
#         self.column_indices = {name: i for i, name in
#                                enumerate(df_train.columns)}

#         # Work out the window parameters.
#         self.input_width = n_prev
#         self.label_width = pred_size
#         self.shift = pred_offset

#         self.total_window_size = n_prev + pred_offset

#         self.input_slice = slice(0, n_prev)
#         self.input_indices = np.arange(self.total_window_size)[
#             self.input_slice]

#         self.label_start = self.total_window_size - self.label_width
#         self.labels_slice = slice(self.label_start, None)
#         self.label_indices = np.arange(self.total_window_size)[
#             self.labels_slice]

#     def split_window(self, features):
#         inputs = features[:, self.input_slice, :]
#         labels = features[:, self.labels_slice, :]
#         if self.label_columns is not None:
#             labels = tf.stack(
#                 [labels[:, :, self.column_indices[name]]
#                     for name in self.label_columns],
#                 axis=-1)

#         # Slicing doesn't preserve static shape information, so set the shapes
#         # manually. This way the `tf.data.Datasets` are easier to inspect.
#         inputs.set_shape([None, self.input_width, None])
#         labels.set_shape([None, self.label_width, None])
#         return inputs, labels

#     def make_dataset(self, data):
#         data = np.array(data, dtype=np.float32)
#         ds = tf.keras.utils.timeseries_dataset_from_array(
#             data=data,
#             targets=None,
#             sequence_length=self.total_window_size,
#             sequence_stride=1,
#             shuffle=True,
#             batch_size=64,)

#         ds = ds.map(self.split_window)

#         return ds

#     def __repr__(self):
#         return '\n'.join([
#             f'Total window size: {self.total_window_size}',
#             f'Input indices: {self.input_indices}',
#             f'Label indices: {self.label_indices}',
#             f'Label column name(s): {self.label_columns}'])

class DatasetManager:

    def __init__(self, df_train: pd.DataFrame, df_test: pd.DataFrame,
                 min_scale: int = 0, max_scale: int = 1, n_prev: int = 96,
                 batch_size: int = 64) -> None:
        # Create self.x_train, y_train, x_test, y_test by extracting "y" column from dataframes
        self.x_train = df_train.drop(columns=['y'])
        self.y_train = df_train['y']
        self.x_test = df_test.drop(columns=['y'])
        self.y_test = df_test['y']
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

    def scale(self) -> None:
        # Scale x and y
        self.x_train = self.x_scaler.transform(self.x_train)
        self.y_train = self.y_scaler.transform(self.y_train)
        self.x_test = self.x_scaler.transform(self.x_test)
        self.y_test = self.y_scaler.transform(self.y_test)

    def invert_scale(self) -> None:
        # Invert scaling for y
        self.y_train = self.y_scaler.inverse_transform(self.y_train)
        self.y_test = self.y_scaler.inverse_transform(self.y_test)

    def get_generator(self) -> TimeseriesGenerator:
        self.scale()
        y_train = self.y_train.reshape(-1)
        y_train = np.insert(y_train, 0, 0)
        y_train = np.delete(y_train, -1)
        generator = TimeseriesGenerator(
            self.x_train, y_train,
            length=self.n_prev, batch_size=self.batch_size)
        return generator