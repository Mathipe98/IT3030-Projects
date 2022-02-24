from stacked_mnist import StackedMNISTData, DataMode
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import InputLayer, Dense, Dropout, Flatten, Reshape, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D
import numpy as np

# class AutoEncoderModel(Model):

#     def __init__(self):
#         super(AutoEncoderModel, self).__init__()
#         encoder = Sequential()
#         encoder.add(InputLayer(input_shape=(28,28,1)))
#         encoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
#         encoder.add(MaxPooling2D((2, 2), padding='same'))
#         encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#         encoder.add(MaxPooling2D((2, 2), padding='same'))
#         encoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#         encoder.add(MaxPooling2D((2, 2), padding='same'))
#         encoder.add(Flatten())
#         encoder.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
#         encoder.add(Dense(10, activation='sigmoid'))
#         self.encoder = encoder
#         # SHAPE UP TO DECODER: (4,4,8)

#         decoder = Sequential()
#         decoder.add(InputLayer(input_shape=(10,)))
#         decoder.add(Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.L1(0.0001)))
#         decoder.add(Reshape((4,4,8)))
#         decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#         decoder.add(UpSampling2D((2, 2)))
#         decoder.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
#         decoder.add(UpSampling2D((2, 2)))
#         decoder.add(Conv2D(16, (3, 3), activation='relu'))
#         decoder.add(UpSampling2D((2, 2)))
#         decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
#         self.decoder = decoder

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

class AutoEncoderModel(Model):
    def __init__(self):
        super(AutoEncoderModel, self).__init__()
        self.encoder = tf.keras.Sequential([
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation='relu',
                   padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
            Conv2D(16, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
            Conv2D(8, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
            Conv2D(2, (3, 3), activation='relu', padding='same', strides=2, kernel_regularizer=tf.keras.regularizers.L1(0.001)),
            Flatten(),
            Dense(2*2*2, activation="sigmoid"),
            ])

        self.decoder = tf.keras.Sequential([
            InputLayer(input_shape=(2*2*2,)),
            Dense(2*2*2, activation="relu"),
            Reshape((2,2,2)),
            ZeroPadding2D(padding=((1,0), (1,0))),
            Conv2DTranspose(
                8, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L1(0.001)),
            ZeroPadding2D(padding=((1,0), (1,0))),
            Conv2DTranspose(
                16, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L1(0.001)),
            Conv2DTranspose(
                32, kernel_size=3, strides=2, activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.L1(0.001)),
            Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
            ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


if __name__ == "__main__":
    asd = AutoEncoderModel()
    asd.encoder.summary()
    asd.decoder.summary()
