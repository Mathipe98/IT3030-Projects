import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Flatten

def get_lstm_model() -> Sequential:
    model = Sequential()
    model.add(LSTM(units=256, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(units=128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(LSTM(units=64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
    return model
    