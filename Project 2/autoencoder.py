import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Tuple
from stacked_mnist import StackedMNISTData, DataMode
from verification_net import VerificationNet
from ae_model import AutoEncoderModel
from tensorflow import keras
from utils import visualize_pictures

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)


class AutoEncoder:

    def __init__(self, generator: StackedMNISTData, force_relearn: bool = False, file_name: str = "./models/ae_model", model: Any = None) -> None:
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
            print(f"AE: Read model from file, so I do not retrain.")
            done_training = True

        except:
            print(f"AE: Could not read weights from file. Must retrain...")
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
        no_channels = x_train.shape[-1]
        filename = './figures/ae_mono_training_results.png' if no_channels == 1 \
            else './figures/ae_colour_training_results.png'
        visualize_pictures(x_train, y_train, decoded_imgs, filename=filename)

    def visualize_testing_results(self) -> None:
        x_test, y_test = self.generator.get_full_data_set(training=False)
        decoded_imgs = self.predict(x_test)
        no_channels = x_test.shape[-1]
        filename = './figures/ae_mono_training_results.png' if no_channels == 1 \
            else './figures/ae_colour_training_results.png'
        visualize_pictures(x_test, y_test, decoded_imgs, filename=filename)

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

    def create_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """This method calculates the mean and standard deviation for each of the 
        32 outputs of the encoder. In other words, the encoder has a latent dim of
        32. For each of these 32 elements, this method calculates the individual mean
        and standard deviation by running the training and testing data through
        the encoder. Thus when generating random samples, the values are extracted
        by randomly getting each value individually by their respective mean/stddev values.

        Raises:
            ValueError: Error to be thrown if model is not yet trained

        Returns:
            Tuple[np.ndarray, np.ndarray]: Arrays containing the mean and stddev for the 32 latent values
        """
        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot generate")
        training_data, _ = self.generator.get_full_data_set(training=True)
        testing_data, _ = self.generator.get_full_data_set(training=False)
        encoder_output_elements = self.model.encoder.layers[-1].output_shape[-1]
        training_encoder_output_shape = (
            training_data.shape[0], encoder_output_elements)
        testing_encoder_output_shape = (
            testing_data.shape[0], encoder_output_elements)
        no_channels = training_data.shape[-1]
        encoded_predictions = np.zeros(shape=training_encoder_output_shape)
        testing_encoded = np.zeros(shape=testing_encoder_output_shape)
        for channel in range(no_channels):
            encoded_training = self.model.encoder.predict(
                training_data[:, :, :, [channel]])
            encoded_predictions += encoded_training
            encoded_testing = self.model.encoder.predict(testing_data[:,:,:,[channel]])
            testing_encoded += encoded_testing
        # Calculate the mean and standard deviation
        mu = np.mean(encoded_predictions, axis=0)
        deviation_1 = testing_encoded - mu
        deviation_2 = abs(deviation_1)**2
        deviation_sum = np.sum(deviation_2, axis=0)
        stddev = np.sqrt(deviation_sum / training_data.shape[0])
        return mu, stddev

    def generate(self, filename) -> None:
        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot generate")
        n = 100
        mu, stddev = self.create_distribution()
        training_data, _ = self.generator.get_full_data_set()
        no_channels = training_data.shape[-1]
        random_data = []
        for _ in range(n):
            random_sample = []
            for j in range(mu.shape[0]):
                colors = []
                for _ in range(no_channels):
                    # Add a random value based on mean and standard deviation of the output of the model
                    random_value = np.random.normal(loc=mu[j], scale=stddev[j]) / no_channels
                    colors.append(random_value)
                random_sample.append(colors)
            random_data.append(random_sample)
        random_data = np.array(random_data)
        decoded_imgs = np.zeros(shape=(n,) + training_data.shape[1:])
        for channel in range(decoded_imgs.shape[-1]):
            decoder_input = random_data[:, :, channel]
            output = self.model.decoder.predict(decoder_input)
            decoded_imgs[:, :, :, channel] += output[:, :, :, 0]
        visualize_pictures(x=decoded_imgs, filename=filename)

    def anomaly_detection(self, filename: str) -> None:
        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot detect anomalies")
        bce = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        testing_data, testing_labels = self.generator.get_full_data_set(
            training=False)
        no_channels = testing_data.shape[-1]
        losses = np.zeros(shape=(testing_data.shape[0], 2))
        for channel in range(no_channels):
            channel_labels = []
            for label in testing_labels:
                if len(str(label)) < channel + 1:
                    channel_labels.append(0)
                else:
                    channel_labels.append(int(str(label)[-1 - channel]) * 10 ** channel)
            channel_labels = np.array(channel_labels)[:, np.newaxis]
            channel_examples = testing_data[:,:,:,[channel]]
            predictions = self.predict(channel_examples)
            prediction_losses = tf.reduce_mean(bce(channel_examples, predictions), axis=[1,2]).numpy()[:, np.newaxis]
            losses_w_labels = np.hstack((prediction_losses, channel_labels))
            losses += losses_w_labels
        sorted_array = losses[losses[:,0].argsort()][::-1]
        anomaly_losses = sorted_array[:20,0]
        anomaly_labels = sorted_array[:20,1].astype(int)
        anomaly_examples = []
        for label in anomaly_labels:
            indeces = np.where(testing_labels == label)
            example_index = indeces[0][0]
            verify_label = testing_labels[example_index]
            assert verify_label == label, "Index of example doesn't match"
            anomaly_examples.append(testing_data[example_index])
        anomaly_examples = np.array(anomaly_examples)
        anomaly_recreations = self.predict(anomaly_examples)
        print(f"Highest losses:\n {anomaly_losses}")
        print(f"Corresponding labels:\n {anomaly_labels}")
        visualize_pictures(anomaly_examples, anomaly_labels, anomaly_recreations, filename=filename)


def showcase_mono() -> None:
    print(f"\n======== SHOWCASING MONO-CHROMATIC IMAGES ========\n")
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    ae = AutoEncoder(generator=gen, force_relearn=False)
    ae.train(epochs=1000)
    print(f"\nVisualizing training images...")
    ae.visualize_training_results()
    print(f"\nVisualizing testing images...\n")
    ae.visualize_testing_results()
    print(f"\nCalculating accuracy..\n")
    ae.test_accuracy(verifier=verifier, tolerance=0.8)
    print(f"\nGenerating images..\n")
    ae.generate(filename='./figures/ae_mono_generation.png')

def showcase_colour() -> None:
    print(f"\n======== SHOWCASING COLOUR IMAGES ========\n")
    gen = StackedMNISTData(
        mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    ae = AutoEncoder(generator=gen, force_relearn=False)
    ae.train(epochs=1000)
    print(f"\nVisualizing training images...")
    ae.visualize_training_results()
    print(f"\nVisualizing testing images...\n")
    ae.visualize_testing_results()
    print(f"\nCalculating accuracy..\n")
    ae.test_accuracy(verifier=verifier, tolerance=0.5)
    print(f"\nGenerating images..\n")
    ae.generate(filename='./figures/ae_colour_generation.png')

def showcase_anomalies() -> None:
    print(f"\n======== SHOWCASING ANOMALY DETECTION ========\n")
    gen1 = StackedMNISTData(
        mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
    gen2 = StackedMNISTData(
        mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=2048)
    ae1 = AutoEncoder(generator=gen1, force_relearn=False, file_name='./models/ae_anomaly_model')
    ae1.train(epochs=1000)
    ae2 = AutoEncoder(generator=gen2, force_relearn=False, file_name='./models/ae_anomaly_model')
    ae2.train(epochs=1000)
    print(f"\nDetecting mono-chromatic anomalies..\n")
    ae1.anomaly_detection(filename='./figures/ae_mono_anomalies.png')
    print(f"\nDetecting colour anomalies..\n")
    ae2.anomaly_detection(filename='./figures/ae_colour_anomalies.png')


if __name__ == "__main__":
    showcase_mono()
    showcase_colour()
    showcase_anomalies()
