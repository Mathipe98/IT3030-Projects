import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from typing import Any
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from tensorflow import keras
from vae_model import VariationalAutoEncoderModel
from utils import visualize_pictures

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


class VAE:

    def __init__(self, generator: StackedMNISTData, force_relearn: bool = False, file_name: str = "./models/vae_model") -> None:
        self.generator = generator
        self.force_relearn = force_relearn
        self.file_name = file_name
        self.model = VariationalAutoEncoderModel()
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=.00025),
                           loss=keras.losses.binary_crossentropy,
                           metrics=['accuracy'], run_eagerly=True)
        self.done_training = False

    def load_weights(self):
        try:
            self.model.load_weights(filepath=self.file_name)
            print(f"VAE: Read model from file, so I do not retrain.")
            done_training = True

        except:
            print(f"VAE: Could not read weights for from file. Must retrain...")
            done_training = False

        return done_training

    def train(self, epochs: int = 100):
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
                monitor='loss', min_delta=0.0001, patience=100, verbose=0, restore_best_weights=True
            )

            for channel in range(x_train_all_channels.shape[-1]):
                # Iterate through all the channels, and train on each channel separately
                x_train = x_train_all_channels[:, :, :, [channel]]
                x_test = x_test_all_channels[:, :, :, [channel]]

                self.model.fit(x=x_train, y=x_train, batch_size=1024, epochs=epochs,
                               validation_data=(x_test, x_test), callbacks=[callback])
            # Save weights and leave
            self.model.save_weights(filepath=self.file_name)
            self.done_training = True
        return self.done_training

    def predict(self, data: np.ndarray) -> np.ndarray:
        no_channels = data.shape[-1]

        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot predict")

        predictions = np.zeros(shape=data.shape)
        for channel in range(no_channels):
            channel_prediction = self.model.predict(data[:, :, :, [channel]])
            predictions[:, :, :, [channel]] += channel_prediction[:, :, :, [0]]

        return predictions
    
    def visualize_training_results(self) -> None:
        x_train, y_train = self.generator.get_full_data_set(training=True)
        decoded_imgs = self.predict(x_train)
        visualize_pictures(x_train, y_train, decoded_imgs)

    def visualize_testing_results(self) -> None:
        x_test, y_test = self.generator.get_full_data_set(training=False)
        decoded_imgs = self.predict(x_test)
        visualize_pictures(x_test, y_test, decoded_imgs)

    def test_accuracy(self, verifier: VerificationNet, tolerance: float = 0.8) -> None:
        x_test, y_test = self.generator.get_full_data_set(training=False)
        images = self.predict(x_test)
        labels = y_test
        cov = verifier.check_class_coverage(data=images, tolerance=tolerance)
        pred, acc = verifier.check_predictability(
            data=images, correct_labels=labels, tolerance=tolerance)
        print(f"Coverage: {100*cov:.2f}%")
        print(f"Predictability: {100*pred:.2f}%")
        print(f"Accuracy: {100 * acc:.2f}%")
    
    def generate(self) -> None:
        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot generate")
        n = 10
        training_data, _ = self.generator.get_full_data_set(training=True)
        no_channels = training_data.shape[-1]
        random_data = K.random_normal(shape=(n,) + self.model.z_layer_input_shape + (no_channels,), mean=0., stddev=1.)
        decoded_imgs = np.zeros(shape=(n,) + training_data.shape[1:])
        for channel in range(decoded_imgs.shape[-1]):
            decoder_input = random_data[:, :, channel]
            output = self.model.decoder.predict(decoder_input)
            decoded_imgs[:, :, :, channel] += output[:, :, :, 0]
        plt.figure(figsize=(20, 4))
        for i in range(1, n + 1):
            ax = plt.subplot(2, n, i)
            plt.imshow(decoded_imgs[i-1])
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def anomaly_detection(self) -> None:
        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot detect anomalies")
        bce = keras.losses.BinaryCrossentropy()
        testing_data, testing_labels = self.generator.get_full_data_set(
            training=False)
        no_channels = testing_data.shape[-1]
        example_loss_map = []
        for i in range(len(testing_data)):
            print(i)
            example = testing_data[i]
            label = testing_labels[i]
            example = example[np.newaxis, :]
            reconstruction = np.zeros(shape=(testing_data.shape[1:]))
            for channel in range(no_channels):
                a = self.model.predict(example[:,:,:,[channel]])
                reconstruction[:,:,[channel]] += a[0,:,:,:]
            loss = bce(example, reconstruction).numpy()
            example_loss_map.append((reconstruction, label, loss))
            if i == 3000:
                break
        sorted_map = sorted(example_loss_map, key=lambda tup: tup[2], reverse=True)
        anomaly_examples = np.array([anomaly[0] for anomaly in sorted_map[:10]])
        anomaly_labels = np.array([anomaly[1] for anomaly in sorted_map[:10]])
        print(f"Highest losses:\n {[anomaly[2] for anomaly in sorted_map[:10]]}")
        print(f"Corresponding labels:\n {[anomaly[1] for anomaly in sorted_map[:10]]}")
        visualize_pictures(anomaly_examples, anomaly_labels, filename='./figures/STACK_anomalies_3000.png')

def showcase_mono() -> None:
    print(f"\n======== SHOWCASING MONO-CHROMATIC IMAGES ========\n")
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    vae = VAE(generator=gen, force_relearn=False)
    vae.train(epochs=2000)
    print(f"\nVisualizing training images...")
    vae.visualize_training_results()
    print(f"\nVisualizing testing images...")
    vae.visualize_testing_results()
    vae.test_accuracy(verifier, tolerance=0.8)
    vae.generate()

def showcase_stack() -> None:
    gen = StackedMNISTData(
        mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    vae = VAE(generator=gen, force_relearn=False)
    vae.train(epochs=2000)
    vae.visualize_training_results()
    vae.visualize_testing_results()
    vae.test_accuracy(verifier, tolerance=0.5)
    vae.generate()

def showcase_anomalies() -> None:
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    vae = VAE(generator=gen, force_relearn=True, file_name='./models/vae_anomaly_model')
    vae.train(epochs=2000)


if __name__ == "__main__":
    showcase_mono()
    showcase_stack()
