import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from tensorflow import keras
from vae_model import VariationalAutoEncoderModel

tf.get_logger().setLevel('ERROR')
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


class VAE:

    def __init__(self, generator: StackedMNISTData, force_relearn: bool = False, file_name: str = "./vae_model/autoencoder_model") -> None:
        self.generator = generator
        self.force_relearn = force_relearn
        self.file_name = file_name
        self.model = VariationalAutoEncoderModel()
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer=keras.optimizers.Adam(learning_rate=.001),
                           metrics=['accuracy'], run_eagerly=True)
        self.done_training = False
    
    def load_weights(self):
        try:
            self.model.load_weights(filepath=self.file_name)
            print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(f"Could not read weights for autoencoder from file. Must retrain...")
            done_training = False

        return done_training

    def train(self, epochs: int=100):
        if not self.force_relearn:
            self.done_training = self.load_weights()
        if self.force_relearn or not self.done_training:
            # Get hold of data
            x_train_all_channels, _ = self.generator.get_full_data_set(
                training=True)
            x_test_all_channels, _ = self.generator.get_full_data_set(
                training=False)
            # Create a callback for stopping if we don't get any progress
            callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=0.0001, patience=25, verbose=0, restore_best_weights=True
            )

            for channel in range(x_train_all_channels.shape[-1]):
                # Iterate through all the channels, and train on each channel separately
                x_train = x_train_all_channels[:, :, :, [channel]]
                x_test = x_test_all_channels[:, :, :, [channel]]

                self.model.fit(x=x_train, y=x_train, batch_size=2048, epochs=epochs,
                               validation_data=(x_test, x_test), callbacks=[callback])
            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True
        return self.done_training

def test():
    print(f"Testing mono accuracy\n")
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    # verifier = VerificationNet(force_learn=False)
    # verifier.train(gen, epochs=100)
    vae = VAE(generator=gen, force_relearn=False)
    vae.train(epochs=100)

if __name__ == "__main__":
    test()
