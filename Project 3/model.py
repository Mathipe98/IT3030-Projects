import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import Sequential
from keras.layers import LSTM, Dense

from data_prep import DatasetManager

class ModelHandler:

    def __init__(self, manager: DatasetManager) -> None:
        self.mng = manager

    def get_model(self) -> Sequential:
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True))
        model.add(LSTM(units=64, return_sequences=True))
        model.add(Dense(units=1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
        return model
    
    def predict_2hrs(self, df: pd.DataFrame, model, start_index: int=0) -> np.ndarray:
        # Add a column to the test-dataframe with the predicted values
        df['previous_y'] = 0
        # Extract resolution and n_prev from self.manager
        resolution = self.mng.resolution
        n_prev = self.mng.n_prev
        # If resolution is 15 minutes, then we need 8*15 = 120 minutes = 2 hrs. Else we need 24 predictions to fill the 2 hours
        if resolution == 15:
            n_preds = 8
        else:
            n_preds = 24
        # Set the previous y value at start index to the model prediction
        results = []
        for i in range(start_index, min(len(df), start_index+n_preds)):
            # Note: use max/min to calculate the correct indices at which to slice the dataframe as well as reshape
            df_slice = df[max(start_index, i-n_prev+1):i+1].to_numpy()
            df_slice = df_slice.reshape(1, min(n_prev, i+1), len(df.columns))
            result = model(df_slice).numpy()[0][0]
            df.loc[i+1, 'previous_y'] = result
            results.append(result)
        return np.array(results)