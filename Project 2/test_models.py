import tensorflow as tf
from tensorflow.keras import layers
from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from keras import Input
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
import numpy as np

class Denoise(Model):
    def __init__(self):
        super(Denoise, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(16, (3, 3), activation='relu',
                          padding='same', strides=2),
            layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])

        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(
                8, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(
                16, kernel_size=3, strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
