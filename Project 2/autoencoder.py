import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Any
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from ae_model import AutoEncoderModel
from tensorflow import keras
from utils import visualize_pictures

tf.get_logger().setLevel('ERROR')
seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


class AutoEncoder:

    def __init__(self, generator: StackedMNISTData, force_relearn: bool = False, file_name: str = "./autoenc_model/autoencoder_model", model: Any = None) -> None:
        self.generator = generator
        self.force_relearn = force_relearn
        self.file_name = file_name
        self.model = model if model is not None else AutoEncoderModel()
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer=keras.optimizers.Adam(learning_rate=.001),
                           metrics=['accuracy'])
        self.distribution = None
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

    def train(self, epochs: int = 10) -> bool:
        """
        Train model if required. As we have a one-channel model we take care to
        only use the first channel of the data.
        """
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
            # Create a function that adds padding to the labels in order to find the correct one by channel

            for channel in range(x_train_all_channels.shape[-1]):
                # Iterate through all the channels, and train on each channel separately
                x_train = x_train_all_channels[:, :, :, [channel]]
                x_test = x_test_all_channels[:, :, :, [channel]]

                # Fit model (same sets for input and output because we want to replicate it)
                self.model.fit(x=x_train, y=x_train, batch_size=2048, epochs=epochs,
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

    def create_distribution(self) -> np.ndarray:
        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot generate")
        training_data, _ = self.generator.get_full_data_set(training=True)
        # For now: only 1 channel
        encoder_output_elements = self.model.encoder.layers[-1].output_shape[-1]
        encoder_output_shape = (
            training_data.shape[0], encoder_output_elements)
        no_channels = training_data.shape[-1]
        encoded_predictions = np.zeros(shape=encoder_output_shape)
        for channel in range(no_channels):
            encoded = self.model.encoder.predict(
                training_data[:, :, :, [channel]]) * 1/no_channels
            encoded_predictions += encoded
        # Take the mean of all the encoder predictions for use in the sampling
        # I.e. we now have the average value of all encodings of all inputs, and we will use this
        # as a basis for drawing random samples for Z
        distribution = np.mean(encoded_predictions, axis=0)
        return distribution

    def generate(self) -> None:
        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot generate")
        n = 10
        distribution = self.create_distribution()
        training_data, _ = self.generator.get_full_data_set()
        no_channels = training_data.shape[-1]
        random_data = []
        for _ in range(n):
            random_sample = []
            for j in range(len(distribution)):
                colors = []
                for _ in range(no_channels):
                    # Create a random value that is based on the mean, where you either subtract or add a chunk of the original value
                    random_value = distribution[j] + np.random.choice(
                        [-1, 1], p=[0.5, 0.5]) * np.random.uniform(low=0.3, high=.4)
                    colors.append(random_value)
                random_sample.append(colors)
            random_data.append(random_sample)
        random_data = np.array(random_data)
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


def test_mono() -> None:
    print(f"Testing mono accuracy\n")
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    ae = AutoEncoder(generator=gen, force_relearn=False)
    ae.train(epochs=1000)
    ae.visualize_training_results()
    ae.visualize_testing_results()
    ae.test_accuracy(verifier=verifier)
    ae.generate()

def test_color() -> None:
    print(f"Testing colour (stack) accuracy\n")
    gen = StackedMNISTData(
        mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    ae = AutoEncoder(generator=gen, force_relearn=False, file_name='./autoenc_model/autoencoder_STACK')
    ae.train(epochs=1000)
    ae.visualize_training_results()
    ae.visualize_testing_results()
    ae.test_accuracy(verifier=verifier, tolerance=0.5)
    ae.generate()

def test_mono_anomaly() -> None:
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    ae = AutoEncoder(generator=gen, force_relearn=False,
                     file_name="./autoenc_model/anomaly_detector")
    ae.train(epochs=500)
    ae.anomaly_detection()

def test_color_anomaly() -> None:
    gen = StackedMNISTData(
        mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    ae = AutoEncoder(generator=gen, force_relearn=False,
                     file_name="./autoenc_model/anomaly_detector")
    ae.train(epochs=500)
    ae.anomaly_detection()


if __name__ == "__main__":
    # test_anomaly_detection()
    test_mono()
    # test_color()
