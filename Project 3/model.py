from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf

from keras import Sequential
from keras.layers import LSTM, Dense


"""
NOTES REGARDING PARAMETERS:


    - Following LSTM units have been tested:
        - [512, 256, 128, 64, 32, 16]

        - The thought of the low-value units was that the model only seems to care much about the
            closest values in time. After much testing, the optimal units values were 128 and 64.

    
    - Following Dense units have been tested:
        - [256, 128, 64, 32, 16, 8]

        - They were tried with different activation functions, including relu, sigmoid, and linear.
        - It seemed that the activation functions in these layers did not improve model performance,
            therefore the linear activation function was used.
    

    - The final LSTM layer was initially used with

                [return_sequences=True],
    
        i.e. output for every input. However this complicated the learning process and resulted in a
        worse model. Therefore it was dropped to just return the final output.
    

    - The learning rate was tested on the following values:
        - [0.0001, 0.001, 0.01, 0.1]

        - A lower learning-rate was tried with a more substantial model, i.e. one with more LSTM and Dense layers.
        - However the addition of layers did not improve the model, therefore the learning rate was kept at 0.001 for the final model
    

    - Finally, an

        L2 REGULARIZATION

    was added to the model. This was done to prevent overfitting.
        - However, it seems that the model actually NEEDS overfitting in order to properly mimic the oscillation found
            during training. Therefore the model tended to 'average' its losses with regularization, resulting in more
            'smooth' curves. Reg was therefore dropped in order for the model to better mimic the oscillation.
"""

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
    