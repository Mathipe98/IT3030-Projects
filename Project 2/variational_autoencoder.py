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
                monitor='loss', min_delta=0.0001, patience=1000, verbose=0, restore_best_weights=True
            )

            for channel in range(x_train_all_channels.shape[-1]):
                # Iterate through all the channels, and train on each channel separately
                x_train = x_train_all_channels[:, :, :, [channel]]
                x_test = x_test_all_channels[:, :, :, [channel]]

                self.model.fit(x=x_train, y=x_train, batch_size=512, epochs=epochs,
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

    def anomaly_detection(self, filename: str) -> None:
        if not self.done_training:
            # Model is not trained yet...
            raise ValueError("Model is not trained; cannot detect anomalies")
        n_random_samples = 1000
        bce = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        testing_data, testing_labels = self.generator.get_full_data_set(
            training=False)
        n_testing_examples = testing_data.shape[0]
        no_channels = testing_data.shape[-1]
        probability_map = np.ones(shape=(n_testing_examples, 2))
        for channel in range(no_channels):
            channel_labels = []
            for label in testing_labels:
                if len(str(label)) < channel + 1:
                    channel_labels.append(0)
                else:
                    channel_labels.append(int(str(label)[-1 - channel]) * 10 ** channel)
            channel_labels = np.array(channel_labels)
            # We need to calculate the predicted outcome of the normal vector
            channel_probabilities = []
            for i in range(testing_data.shape[0]):
                print(i)
                normal_vector = K.random_normal(shape=(n_random_samples,) + self.model.z_layer_input_shape, mean=0., stddev=1.)
                normal_vector_outputs = self.model.decoder.predict(normal_vector)
                image = testing_data[i]
                label = testing_labels[i]
                # Shape is 28x28x1n_testing_examples
                image_repeated = np.repeat(image.reshape(-1, *image.shape), n_random_samples, axis=0).astype(float)
                logpx_z = -bce(normal_vector_outputs, image_repeated)
                px_z = np.exp(logpx_z)
                px = np.mean(px_z)
                channel_probabilities.append(px)
            channel_probabilities = np.array(channel_probabilities)
            probability_map[:,0] *= channel_probabilities
            probability_map[:,1] += channel_labels
        sorted_array = probability_map[probability_map[:,0].argsort()]
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
        print(f"Lowest probabilities:\n {anomaly_losses}")
        print(f"Corresponding labels:\n {anomaly_labels}")
        visualize_pictures(anomaly_examples, anomaly_labels, anomaly_recreations, filename=filename)

def showcase_mono() -> None:
    print(f"\n======== SHOWCASING MONO-CHROMATIC IMAGES ========\n")
    gen = StackedMNISTData(
        mode=DataMode.MONO_BINARY_COMPLETE, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    vae = VAE(generator=gen, force_relearn=False)
    vae.train(epochs=1000)
    print(f"\nVisualizing training images...")
    vae.visualize_training_results()
    print(f"\nVisualizing testing images...\n")
    vae.visualize_testing_results()
    print(f"\nCalculating accuracy..\n")
    vae.test_accuracy(verifier=verifier, tolerance=0.8)
    print(f"\nGenerating images..\n")
    vae.generate()

def showcase_colour() -> None:
    print(f"\n======== SHOWCASING COLOUR IMAGES ========\n")
    gen = StackedMNISTData(
        mode=DataMode.COLOR_BINARY_COMPLETE, default_batch_size=2048)
    verifier = VerificationNet(force_learn=False)
    verifier.train(gen, epochs=100)
    vae = VAE(generator=gen, force_relearn=False)
    vae.train(epochs=1000)
    print(f"\nVisualizing training images...")
    vae.visualize_training_results()
    print(f"\nVisualizing testing images...\n")
    vae.visualize_testing_results()
    print(f"\nCalculating accuracy..\n")
    vae.test_accuracy(verifier=verifier, tolerance=0.5)
    print(f"\nGenerating images..\n")
    vae.generate()

def showcase_anomalies() -> None:
    print(f"\n======== SHOWCASING ANOMALY DETECTION ========\n")
    gen1 = StackedMNISTData(
        mode=DataMode.MONO_BINARY_MISSING, default_batch_size=2048)
    gen2 = StackedMNISTData(
        mode=DataMode.COLOR_BINARY_MISSING, default_batch_size=2048)
    vae1 = VAE(generator=gen1, force_relearn=False, file_name='./models/vae_anomaly_model')
    vae1.train(epochs=10000)
    vae2 = VAE(generator=gen2, force_relearn=False, file_name='./models/vae_anomaly_model')
    vae2.train(epochs=10000)
    print(f"\nDetecting mono-chromatic anomalies..\n")
    vae1.anomaly_detection(filename='./figures/vae_mono_anomalies.png')
    print(f"\nDetecting colour anomalies..\n")
    #vae2.anomaly_detection(filename='./figures/vae_colour_anomalies.png')


if __name__ == "__main__":
    # showcase_mono()
    # showcase_stack()
    showcase_anomalies()
