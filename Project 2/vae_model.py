import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential, Model
from keras.layers import (InputLayer, Dense, Dropout, Flatten, Reshape, Conv2D, Lambda, Input,
                        Conv2DTranspose, Cropping2D, ZeroPadding2D, BatchNormalization, LeakyReLU)

loss_tracker = tf.keras.metrics.Mean(name="loss")

class VariationalAutoEncoderModel(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_input_shape = (28,28,1)
        self.z_layer_input_shape = (16,)
        self.decoder_input_shape = (16,)
        self.encoder = self.create_encoder(input_shape=self.encoder_input_shape)
        self.z_latent = self.create_z_layer()
        self.decoder = self.create_decoder(input_shape=self.decoder_input_shape)
    
    @tf.function
    def train_step(self, data):

        x, y = data

        with tf.GradientTape(persistent=True) as tape:
            mean, log_variance = self.encoder(x, training=True)
            z_latent = self.z_latent([mean, log_variance])
            y_pred = self.decoder(z_latent, training=True)
            loss = self.vae_loss(y_true=tf.cast(x, dtype="float32"), y_pred=y_pred, mean=mean, log_variance=log_variance)

        encoder_variables = self.encoder.trainable_variables
        decoder_variables = self.decoder.trainable_variables
        encoder_gradients = tape.gradient(loss, encoder_variables)
        decoder_gradients = tape.gradient(loss, decoder_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(encoder_gradients, encoder_variables))
        self.optimizer.apply_gradients(zip(decoder_gradients, decoder_variables))

        # Update metrics
        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}
    
    @property
    def metrics(self):
        return [loss_tracker]
    
    @tf.function
    def call(self, x):
        encoded = self.encoder(x)
        z = self.z_latent(encoded)
        decoded = self.decoder(z)
        return decoded
    
    def KL(self, Z_sigma: np.ndarray, Z_mean: np.ndarray) -> float:
        # Follow the following formula for KL divergence:
        # KL = 1/2 * [ sum{ -(log(variance) + 1) + variance^2 + mean^2 } ]
        # Link 1: https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048
        # Where KL also corresponds to: KL = log[ p(z) ] - log[ q(z | x) ]
        # Link 2: https://stats.stackexchange.com/questions/304289/variational-autoencoder-understanding-the-latent-loss?rq=1
        # Which is the formula used in Tensorflow example:
        # Link 3: https://www.tensorflow.org/tutorials/generative/cvae
        
        variance = Z_sigma ** 2 # sigma^2
        log_variance = K.log(variance) # log(sigma^2)
        mean_squared = K.square(Z_mean) # mu^2
        loss = 0.5 * K.sum(variance + mean_squared - log_variance - 1, axis=1)
        return loss
    
    def RL(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        bce = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        loss = bce(y_true, y_pred)
        reduced_sum = -tf.reduce_sum(loss, axis=[1,2])
        return reduced_sum

    
    def vae_loss(self, y_true: np.ndarray, y_pred: np.ndarray, mean: np.ndarray, log_variance: np.ndarray) -> np.ndarray:
        """This function calculates the ELBO loss, which consists of:
        ELBO = L(q) = -KL(q(z|x) || p(z)) + E[ log(p(x | z)) ].
        Notes to self:
        p(x|z) = output of decoder, so to get this distribution we simply pass
        an input through the entire network.
        p(z|x) ~= q(z|x) = output of encoder
        p(z) = normal distribution (because we assume it)


        Args:
            y_true (np.ndarray): True y-values
            y_pred (np.ndarray): predicted y-values
            mean (np.ndarray): Mean output of the encoder
            log_variance (np.ndarray): Log variance output of the encoder

        Returns:
            np.ndarray: Loss value
        """
        r_loss = self.RL(y_true, y_pred)
        kl_loss = self.KL(mean, log_variance)
        return -(-kl_loss + r_loss)

    def create_encoder(self, input_shape: tuple) -> Model:
        inputs = Input(shape=input_shape, name='input_layer')
        x = Conv2D(16, kernel_size=3, strides=2, padding="same", activation="relu", name="conv_1")(inputs)
        x = Conv2D(32, kernel_size=3, strides=2, padding="same", activation="relu", name="conv_2")(x)
        flatten = Flatten()(x)

        mean = Dense(self.z_layer_input_shape[0], name='mean')(flatten)
        log_variance = Dense(self.z_layer_input_shape[0], name='log_variance')(flatten)

        model = Model(inputs, (mean, log_variance), name="Encoder")
        return model
    
    def create_decoder(self, input_shape: tuple):
        inputs = Input(shape=input_shape, name='input_layer')
        x = Dense(7*7*32, activation="relu", name='dense_1')(inputs)
        x = Reshape((7, 7, 32), name='reshape_layer')(x)
        x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation="relu", name='conv_transpose_1')(x)
        x = Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation="relu", name='conv_transpose_2')(x)
        outputs = Conv2DTranspose(1, kernel_size=3, padding='same', activation="sigmoid", name='conv_transpose_3')(x)

        model = Model(inputs, outputs, name="Decoder")
        return model
    
    def create_z_layer(self) -> Model:
        mean = Input(shape=self.z_layer_input_shape, name="mean_input")
        log_variance = Input(shape=self.z_layer_input_shape, name="log_variance_input")
        sampled_z_layer = Lambda(self.sample_z_layer, name="encoder_output")([mean, log_variance])
        encoder_output_model = Model([mean, log_variance], sampled_z_layer, name="Latent_Vector")
        return encoder_output_model
    
    def sample_z_layer(self, distribution_params: tuple):
        """This method works as a way to create a deterministic Z-layer (latent space)
        by drawing values from deterministic vectors with mu and sigma, while incorporating
        an epsilon which is drawn randomly from a normal distribution N(0,1).

        Args:
            distribution_params (tuple): The outputs of the vectors with mu and sigma, i.e. their dimensions

        Returns:
            float: A value which is a combination of the inputs and the random variable epsilon
        """
        mean, log_variance = distribution_params
        epsilon = K.random_normal(shape=K.shape(mean), mean=0., stddev=1.)
        return mean + K.exp(log_variance / 2) * epsilon

if __name__ == "__main__":
    test = VariationalAutoEncoderModel()
    test.encoder.summary()
    test.z_latent.summary()
    test.decoder.summary()