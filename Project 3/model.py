import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten

def get_lstm_model() -> Sequential:
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, dropout=0.1))
    model.add(LSTM(units=64, return_sequences=False, dropout=0.1))
    model.add(Dense(64))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['mean_absolute_error'])
    return model
    