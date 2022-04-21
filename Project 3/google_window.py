import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import numpy as np
import pandas as pd
import tensorflow as tf

from model import get_lstm_model


class WindowGenerator():
    def __init__(self, n_prev: int, n_preds: int, resolution: int,
                 start_index: int = 1, batch_size: int = 32, target: str = "y"):
        if n_preds > batch_size:
            raise ValueError("n_preds must be <= batch_size; cannot sample from batch larger than number of samples.")
        self.n_prev = n_prev
        self.n_preds = n_preds
        self.resolution = resolution
        self.start_index = start_index
        self.batch_size = batch_size
        self.target = target

    def make_dataset(self, data: pd.DataFrame, training: bool = True) -> tf.data.Dataset:
        if training:
            x = data.drop(columns=self.target)
            y = data[self.target]
            # Extract all rows apart from the very last one (because we don't have its "correct" value)
            features = np.array(x, dtype=np.float32)[:-1]
            labels = np.array(y, dtype=np.float32)
            labels = np.roll(labels, shift=-idx2, axis=0)[:-1]
        else:
            idx1 = max(self.start_index - self.n_prev, 0)
            idx2 = min(self.start_index, len(data))
            # Print indeces
            print(f"idx1: {idx1}, idx2: {idx2}")
            x = data.drop(columns=self.target)[idx1:idx2]
            print(f"Resulting x: {x}")
            features = np.array(x, dtype=np.float32)
            labels = None
            print(f"Actual target with these indices: {data[self.target][idx2]}")
        print(features.shape)
        # Roll the labels array s.t. they (timewise) match the inputs, i.e. that features
        # contain examples t_0 -> t_k, and y then contains t_k+1
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=features,
            targets=labels,
            sequence_length=self.n_prev,
            sampling_rate=1,
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
    w2 = WindowGenerator(n_prev=2, n_preds=1, resolution=5,
                         start_index=10, batch_size=2, target='y')
    print(f"df_val index 10 answer: {df_val['y'][10]}")
    tf_dataset = w2.make_dataset(df_val, training=False)
    print(tf_dataset)

    model = get_lstm_model()
    # model.evaluate(tf_dataset)
