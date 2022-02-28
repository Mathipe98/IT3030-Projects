from stacked_mnist import StackedMNISTData, DataMode
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import (InputLayer, Dense, Dropout, Flatten, Reshape, Conv2D,
                        Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D,
                        ZeroPadding2D, Normalization)
import numpy as np

class AutoEncoderModel(Model):

    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(28,28,1)))
        encoder.add(ZeroPadding2D(padding=((4,0), (4,0))))
        encoder.add(Conv2D(64, (3,3), activation="relu", padding="same", strides=2,))
        encoder.add(Conv2D(32, (3,3), activation="relu", padding="same", strides=2,))
        encoder.add(Conv2D(16, (3,3), activation="relu", padding="same",  strides=2,))
        encoder.add(Conv2D(8, (3,3), activation="relu", padding="same",  strides=2,))
        encoder.add(Flatten())
        self.encoder = encoder

        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(self.encoder.layers[-1].output_shape[1:])))
        decoder.add(Reshape((self.encoder.layers[-2].output_shape[1:])))
        decoder.add(Conv2DTranspose(8, (3,3), activation="relu", padding="same",  strides=2,))
        decoder.add(Conv2DTranspose(16, (3,3), activation="relu", padding="same",  strides=2,))
        decoder.add(Conv2DTranspose(32, (3,3), activation="relu", padding="same", strides=2,))
        decoder.add(Conv2DTranspose(64, (3,3), activation="relu", padding="same", strides=2,))
        decoder.add(Conv2D(1, (3,3), padding="same", activation="sigmoid"))
        decoder.add(Cropping2D(cropping=((4,0), (4,0))))
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    asd = AutoEncoderModel()
    asd.encoder.summary()
    asd.decoder.summary()
