from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D
import numpy as np


class AutoEncoder(Model):

    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()

        encoder = Sequential()
        encoder.add(Conv2D(32, kernel_size=(3, 3), strides=2, activation='relu', padding="same", input_shape=(28, 28, 1)))
        for _ in range(3):
            encoder.add(Conv2D(64, kernel_size=(3, 3), strides=2, activation='relu', padding="same"))
            encoder.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
            encoder.add(Dropout(0.25))
        encoder.add(Flatten())
        encoder.add(Dense(128, activation='relu'))
        encoder.add(Dropout(0.5))
        encoder.add(Dense(10, activation='softmax'))
        self.encoder = encoder

        # In the first iteration, let's just try to reverse the entire model as-is
        decoder = Sequential()
        decoder.add(Dropout(0.5))
        decoder.add(Dense(128, activation="relu"))
        decoder.add(Flatten())
        for _ in range(3):
            decoder.add(Dropout(0.25))
            decoder.add(Conv2DTranspose(64, kernel_size=(3,3), strides=2, activation="relu", padding="same"))
        decoder.add(Conv2D(1, kernel_size=(3,3), activation="sigmoid", padding="same"))
        self.decoder = decoder
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    autoencoder = AutoEncoder()
    autoencoder.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=keras.optimizers.Adam(learning_rate=.01),
                      metrics=['accuracy'])
    autoencoder.summary()