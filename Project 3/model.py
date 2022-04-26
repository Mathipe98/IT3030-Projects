from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Sequential
from keras.layers import LSTM, Dense

# Note to self:
#   Best parameters for y-pred:
#    LSTM_units = [128, 64]
#    Dense_units = [64, 32]
# NO DROPOUT

def get_model(lstm_units: List[int], dense_units: List[int], lr: float) -> Sequential:
    model = Sequential()
    for i in lstm_units:
        if i == lstm_units[-1]:
            seq = False
        else:
            seq = True
        model.add(LSTM(units=i, return_sequences=seq, dropout=0.1))
    for i in dense_units:
        model.add(Dense(i))
    model.add(Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(loss='mse', optimizer=opt, metrics=['mean_absolute_error'])
    return model
    