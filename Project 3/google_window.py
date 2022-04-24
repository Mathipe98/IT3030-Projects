import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import matplotlib.pyplot as plt

from agent import Agent
from model import get_lstm_model


class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[
            self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[
            self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                    for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='T (degC)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=1)
        
        # for thing in ds:
        #     asd = self.split_window(thing)

        ds = ds.map(self.split_window)

        # print(f"RESULTING DATASET FROM MAKE FUNC: {ds}")
        # for a,b in ds:
        #     print("Printing first 1 examples")
        #     a = a.numpy()
        #     b = b.numpy()
        #     print(f"\na shape: {a.shape}")
        #     print(f"b shape: {b.shape}\n\n")
        #     break
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)


    @property
    def val(self):
        return self.make_dataset(self.val_df)


    @property
    def test(self):
        print("CALLING TEST PROPERTY")
        return self.make_dataset(self.test_df)


    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


if __name__ == '__main__':
    agent = Agent(n_prev=3, start_index=0, batch_size=1, resolution=5)
    df_train = pd.read_csv(
        './Project 3/no1_train.csv').drop("start_time", axis=1)
    df_val = pd.read_csv(
        './Project 3/no1_validation.csv').drop("start_time", axis=1)
    # Add column to df_train and df_val that are just incremental counters
    df_train['counter'] = df_train.index
    df_val['counter'] = df_val.index
    agent.fit_scalers_to_df(df_train)
    scaled_train = agent.transform_df(df_train)
    scaled_test = agent.transform_df(df_val)
    print(f"First 15 rows of df_val:\n{scaled_test.head(15)}")
    # window = WindowGenerator(input_width=3, label_width=1, shift=0, train_df=scaled_train, val_df=None, test_df=scaled_test, label_columns=['y'])
    # ds1 = window.train
    k = 0
    ds2 = agent.make_testing_dataset(df_val.drop(columns='y'))
    for a in ds2:
        # a_inp, a_out = a
        a_inp = a
        a_inp = a_inp.numpy()
        # a_out = a_out.numpy()
        print(f"\na_inp shape: {a_inp.shape}")
        for i in range(a_inp.shape[0]):
            curr_inp = a_inp[i]
            # Make a dataframe with columns from df_train except "y"
            curr_df = pd.DataFrame(curr_inp, columns=df_val.drop(columns="y").columns)
            print(f"Training X turned into df:\n{curr_df}")
            # print(f"Answer to above prediction: {a_out[i]}")
            # print(f"Window X turned into df:\n{pd.DataFrame(win_inp[i], columns=df_train.columns)}")
            # print(f"Answer to above prediction: {win_out[i]}")
        print(f"Shape of model output: {get_lstm_model()(a_inp).numpy().shape}")
        if k >= 3:
            break
        k += 1
    model = get_lstm_model()
    # agent.train(scaled_train)