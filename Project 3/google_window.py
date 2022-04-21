import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import tensorflow as tf

from model import get_lstm_model


class WindowGenerator():
    def __init__(self, n_prev: int, n_preds: int, resolution: int,
                 start_index: int = 1, batch_size: int = 32, target: str = "y"):
        self.n_prev = n_prev
        self.n_preds = n_preds
        self.resolution = resolution
        self.start_index = start_index
        self.batch_size = batch_size
        self.target = target
    
    def make_training_dataset(self, data: pd.DataFrame) -> tf.data.Dataset:
        print("Generating training dataset, therefore start_index has no effect.")
        x = data.drop(columns=self.target)
        y = data[self.target]
        # Extract all rows apart from the very last one (because we don't have its "correct" value)
        features = np.array(x, dtype=np.float32)[:-1]
        labels = np.array(y, dtype=np.float32)
        labels = np.roll(labels, shift=-self.n_prev, axis=0)[:-1]
        # Roll the labels array s.t. they (timewise) match the inputs, i.e. that features
        # contain examples t_0 -> t_k, and y then contains t_k+1
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

    def __repr__(self):
        return ''


if __name__ == '__main__':
    df_train = pd.read_csv(
        './Project 3/no1_train.csv').drop("start_time", axis=1)
    df_val = pd.read_csv(
        './Project 3/no1_validation.csv').drop("start_time", axis=1)
    print(df_val.head(11))
    w2 = WindowGenerator(n_prev=50, n_preds=50, resolution=5,
                         start_index=50, batch_size=64, target='y')
    print(f"df_val index 10 answer: {df_val['y'][10]}")
    training_ds = w2.make_training_dataset(df_train[0:100])
    testing_ds = w2.make_testing_dataset(df_val)
    model = get_lstm_model()
    print(list(testing_ds))
    # model.fit(training_ds, epochs=1)
    for val_tensor in testing_ds:
        print("H")
        print(val_tensor.shape)
        # print(b.shape)
        result = model(val_tensor).numpy().shape
        print(f"Result: {result}")
        break
