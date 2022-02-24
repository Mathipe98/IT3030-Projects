from stacked_mnist import StackedMNISTData, DataMode
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import InputLayer, Dense, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
import numpy as np

class AutoEncoderModel(Model):

    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(28,28,1)))
        encoder.add(Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'))
        encoder.add(Conv2D(16, (3, 3), activation='relu', strides=2, padding='same'))
        encoder.add(Flatten())
        # Shape is now 7 * 7 * 16
        self.encoder = encoder
        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(7*7*16,)))
        decoder.add(Reshape((7,7,16)))
        decoder.add(Conv2DTranspose(16, (3,3), strides=2, padding="same"))
        decoder.add(Conv2DTranspose(32, (3,3), strides=2, padding="same"))
        decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    asd = AutoEncoderModel()
    asd.encoder.summary()
    asd.decoder.summary()
