from stacked_mnist import StackedMNISTData, DataMode
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import (InputLayer, Dense, Dropout, Flatten, Reshape, Conv2D,
                        Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D,
                        ZeroPadding2D, Normalization)
import numpy as np


# class AutoEncoderModel(Model):

#     def __init__(self):
#         super(AutoEncoderModel, self).__init__()
#         encoder = Sequential()
#         encoder.add(InputLayer(input_shape=(28, 28, 1)))
#         encoder.add(Conv2D(32, (3, 3), activation='relu',
#                     strides=2, padding='same'))
#         encoder.add(Conv2D(16, (3, 3), activation='relu',
#                     strides=2, padding='same'))
#         encoder.add(Flatten())
#         encoder.add(Dense(7*7*16, activation="relu"))
#         encoder.add(Dropout(0.2))
#         #encoder.add(Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
#         encoder.add(Dense(10, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
#         self.encoder = encoder
#         self.encoder.compile(loss=keras.losses.binary_crossentropy,
#                            optimizer=keras.optimizers.Adam(learning_rate=.01),
#                            metrics=['accuracy'])
#         # Shape is now 7 * 7 * 16
#         decoder = Sequential()
#         decoder.add(InputLayer(input_shape=(10,)))
#         #decoder.add(Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
#         decoder.add(Dense(7*7*16, activation="relu", kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
#         decoder.add(Reshape((7, 7, 16)))
#         decoder.add(Conv2DTranspose(16, (3, 3), activation="relu", strides=2, padding="same"))
#         decoder.add(Conv2DTranspose(32, (3, 3), activation="relu", strides=2, padding="same"))
#         decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
#         self.decoder = decoder
#         self.decoder.compile(loss=keras.losses.binary_crossentropy,
#                            optimizer=keras.optimizers.Adam(learning_rate=.01),
#                            metrics=['accuracy'])

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


# class AutoEncoderModel(Model):

#     def __init__(self):
#         super(AutoEncoderModel, self).__init__()
#         encoder = Sequential()
#         encoder.add(Flatten())
#         encoder.add(InputLayer(input_shape=(28,28,1)))
#         encoder.add(Dense(28*28*1, activation="relu"))
#         encoder.add(Dense(128, activation="relu"))
#         encoder.add(Dense(50, activation="relu"))
#         encoder.add(Dense(10, activation="softmax"))
#         self.encoder = encoder
#         self.encoder.compile(loss=keras.losses.binary_crossentropy,
#                            optimizer=keras.optimizers.Adam(learning_rate=.01),
#                            metrics=['accuracy'])

#         decoder = Sequential()
#         decoder.add(InputLayer(input_shape=(10,)))
#         decoder.add(Dense(128, activation="relu"))
#         decoder.add(Dense(28*28*1, activation="relu"))
#         decoder.add(Reshape((28,28,1)))
#         self.decoder = decoder
#         self.decoder.compile(loss=keras.losses.binary_crossentropy,
#                            optimizer=keras.optimizers.Adam(learning_rate=.01),
#                            metrics=['accuracy'])


#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded


class AutoEncoderModel(Model):

    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        relu_initializer = tf.keras.initializers.VarianceScaling(
                             scale=0.1, mode='fan_in', distribution='uniform')
        # sigmoid_initializer = tf.keras.initializers.GlorotUniform()
        regularizer = tf.keras.regularizers.L1(0.001)
        encoder = Sequential()
        encoder.add(InputLayer(input_shape=(28,28,1)))
        encoder.add(ZeroPadding2D(padding=((4,0), (4,0))))
        # encoder.add(Dropout(0.5))
        encoder.add(Conv2D(64, (3,3), activation="relu", padding="same", strides=2,))# kernel_initializer=relu_initializer))
        encoder.add(Conv2D(32, (3,3), activation="relu", padding="same", strides=2,))# kernel_initializer=relu_initializer))
        encoder.add(Conv2D(16, (3,3), activation="relu", padding="same",  strides=2,))# kernel_initializer=relu_initializer))
        encoder.add(Conv2D(8, (3,3), activation="relu", padding="same",  strides=2,))# kernel_initializer=relu_initializer))
        encoder.add(Flatten())
       # encoder.add(Dense(7*7*1, activation="sigmoid"))
        # encoder.add(Dense(15, activation="sigmoid"))
        self.encoder = encoder

        decoder = Sequential()
        decoder.add(InputLayer(input_shape=(self.encoder.layers[-1].output_shape[1:])))
        #decoder.add(Dense(200, activation="relu"))
        #decoder.add(Dense(7*7*1, activation="sigmoid"))
        decoder.add(Reshape((self.encoder.layers[-2].output_shape[1:])))
        decoder.add(Conv2DTranspose(8, (3,3), activation="relu", padding="same",  strides=2,))# kernel_initializer=relu_initializer))
        decoder.add(Conv2DTranspose(16, (3,3), activation="relu", padding="same",  strides=2,))# kernel_initializer=relu_initializer))
        decoder.add(Conv2DTranspose(32, (3,3), activation="relu", padding="same", strides=2,))# kernel_initializer=relu_initializer))
        decoder.add(Conv2DTranspose(64, (3,3), activation="relu", padding="same", strides=2,))# kernel_initializer=relu_initializer))
        decoder.add(Conv2D(1, (3,3), padding="same", activation="sigmoid"))
        decoder.add(Cropping2D(cropping=((4,0), (4,0))))
        # decoder.add(Conv2D(1, (3,3), activation="sigmoid", padding="same"))
        self.decoder = decoder

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    asd = AutoEncoderModel()
    asd.encoder.summary()
    asd.decoder.summary()
