import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from autoencoder_model import AutoEncoderModel
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, Conv2DTranspose, MaxPooling2D

from test_models import Denoise
tf.get_logger().setLevel('ERROR')

def visualize_pictures(x_train: np.ndarray, y_train: np.ndarray, decoded_imgs: np.ndarray) -> None:
    n = 10
    random_start = np.random.randint(0, x_train.shape[0] - 11)
    plt.figure(figsize=(20, 4))
    for i in range(1, n + 1):
        # Display original
        ax = plt.subplot(2, n, i)
        original = x_train[i + random_start].astype(np.float64)
        plt.imshow(original)
        plt.title(y_train[i + random_start])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(2, n, i + n)
        copy = decoded_imgs[i + random_start]
        plt.imshow(copy)
        plt.title(y_train[i + random_start])
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()





class AutoEncoder:

    def __init__(self, force_relearn: bool = False, file_name: str = "./autoenc_model/autoencoder_model", model: Any = None) -> None:
        self.force_relearn = force_relearn
        self.file_name = file_name
        self.model = model if model is not None else AutoEncoderModel()
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer=keras.optimizers.Adam(learning_rate=.01),
                           metrics=['accuracy'])

    def load_weights(self):
        try:
            self.model.load_weights(filepath=self.file_name)
            print(f"Read model from file, so I do not retrain")
            done_training = True

        except:
            print(f"Could not read weights for autoencoder from file. Must retrain...")
            done_training = False

        return done_training

    def train(self, generator: StackedMNISTData, epochs: int = 10) -> bool:
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
        if not self.force_relearn:
            self.done_training = self.load_weights()
        if self.force_relearn or not self.done_training:
            # Get hold of data
            x_train_all_channels, _ = generator.get_full_data_set(
                training=True)
            x_test_all_channels, _ = generator.get_full_data_set(
                training=False)
            
            # Create a callback for early stopping to avoid overfit
            callback = tf.keras.callbacks.EarlyStopping(
                monitor='loss', min_delta=0.0001, patience=25, verbose=0, restore_best_weights=True
            )
            for channel in range(x_train_all_channels.shape[-1]):
                # Iterate through all the channels, and train on each channel separately
                x_train = x_train_all_channels[:, :, :, [channel]]
                x_test = x_test_all_channels[:, :, :, [channel]]
                # Fit model (same sets for input and output because we want to replicate it)
                self.model.fit(x=x_train, y=x_train, batch_size=1024, epochs=epochs,
                               validation_data=(x_test, x_test),) #callbacks=[callback])
            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True

        return self.done_training

    def predict(self, data: np.ndarray) -> np.ndarray:
        no_channels = data.shape[-1]

        if not self.done_training:
            # Model is not trained yet...
            raise ValueError(
                "Model is not trained, so makes no sense to try to use it")

        predictions = np.zeros(shape=data.shape)
        for channel in range(no_channels):
            channel_prediction = self.model.predict(data[:, :, :, [channel]])
            predictions[:, :, :, channel] += channel_prediction[:, :, :, 0]

        return predictions

    def visualize_training_results(self, generator: StackedMNISTData) -> None:
        x_train, y_train = generator.get_full_data_set(training=True)
        decoded_imgs = self.predict(x_train)
        visualize_pictures(x_train, y_train, decoded_imgs)

    def test_accuracy(self, generator: StackedMNISTData, verifier: VerificationNet, tolerance: float=0.8) -> None:
        x_test, y_test = generator.get_full_data_set(training=False)

        images = self.predict(x_test)
        labels = y_test
        cov = verifier.check_class_coverage(data=images, tolerance=tolerance)
        pred, acc = verifier.check_predictability(
            data=images, correct_labels=labels, tolerance=tolerance)
        print(f"Coverage: {100*cov:.2f}%")
        print(f"Predictability: {100*pred:.2f}%")
        print(f"Accuracy: {100 * acc:.2f}%")
    
    def generate(self, color: str="mono") -> None:
        if color == "mono":
            encoding_dim = self.model.encoder.layers[-1].output_shape[1:]
        n = 10
        data = np.random.randn(n, *encoding_dim)
        # data = np.random.uniform(size=(n,) + encoding_dim)
        decoded_imgs = self.model.decoder.predict(data)
        plt.figure(figsize=(20, 4))
        for i in range(1, n + 1):
            ax = plt.subplot(2, n, i)
            plt.imshow(decoded_imgs[i-1])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()


def train_new_model():
    ae = AutoEncoder(force_relearn=True)
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    ae.train(gen, epochs=10000)
    ae.model.summary()

def test_autoencoder_accuracy() -> None:
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)

    ae = AutoEncoder(force_relearn=False)
    ae.train(generator=gen, epochs=200)

    ae.visualize_training_results(gen)

    ae.test_accuracy(generator=gen, verifier=verifier)

def test_generation() -> None:

    ae = AutoEncoder(force_relearn=False)
    ae.train(generator=None, epochs=200)
    ae.generate(color="mono")
    # ae.model.encoder.summary()
    # ae.model.decoder.summary()




if __name__ == "__main__":
    train_new_model()
    test_autoencoder_accuracy()
    test_generation()
