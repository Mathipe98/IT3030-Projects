import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
from stacked_mnist import StackedMNISTData, DataMode
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential, Model
from keras.layers import (InputLayer, Dense, Dropout, Flatten, Reshape, Conv2D, Lambda, Input,
                        Conv2DTranspose, Cropping2D, ZeroPadding2D, BatchNormalization, LeakyReLU)


class VariationalAutoEncoderModel(Model):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_input_shape = (28,28,1)
        self.z_layer_input_shape = (2,)
        self.decoder_input_shape = (2,)
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
            loss = self.vae_loss(x, y_pred, mean, log_variance)

        encoder_variables = self.encoder.trainable_variables
        decoder_variables = self.decoder.trainable_variables
        encoder_gradients = tape.gradient(loss, encoder_variables)
        decoder_gradients = tape.gradient(loss, decoder_variables)
        
        # Update weights
        self.optimizer.apply_gradients(zip(encoder_gradients, encoder_variables))
        self.optimizer.apply_gradients(zip(decoder_gradients, decoder_variables))

        # Update metrics
        self.compiled_metrics.update_state(x, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def call(self, x):
        encoded = self.encoder(x)
        z = self.z_latent(encoded)
        decoded = self.decoder(z)
        return decoded
    
    def KL(self, Z_sigma: np.ndarray, Z_mean: np.ndarray) -> float:
        kl_loss =  -0.5 * K.sum(1 + K.log(Z_sigma ** 2) - K.square(Z_mean) - Z_sigma ** 2, axis = 1)
        return kl_loss
    
    def RL(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        bce = keras.losses.BinaryCrossentropy()
        return K.exp(bce(y_true, y_pred))
    
    def vae_loss(self, y_true, y_pred, mean, log_variance):
        r_loss = self.RL(y_true, y_pred)
        kl_loss = self.KL(mean, log_variance)
        return  r_loss + kl_loss
    
    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    @tf.function
    def compute_loss(self, x, y, y_pred, sample_weight):
        mean, logvar = self.encoder(x)
        loss = self.vae_loss(y_true=y, y_pred=y_pred, mean=mean, log_variance=logvar)
        # z = self.sample_z_layer(mean, logvar)
        # x_logit = model.decode(z)
        # cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        # logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        # logpz = self.log_normal_pdf(z, 0., 0.)
        # logqz_x = self.log_normal_pdf(z, mean, logvar)
        return loss # -tf.reduce_mean(logpx_z + logpz - logqz_x)

    def create_encoder(self, input_shape: tuple) -> Model:
        
        inputs = Input(shape=input_shape, name='input_layer')
        x = Conv2D(32, kernel_size=3, strides= 1, padding='same', name='conv_1')(inputs)
        x = BatchNormalization(name='bn_1')(x)
        x = LeakyReLU(name='lrelu_1')(x)
        
        
        x = Conv2D(64, kernel_size=3, strides= 2, padding='same', name='conv_2')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = LeakyReLU(name='lrelu_2')(x)
        
        
        x = Conv2D(64, kernel_size=3, strides=2, padding='same', name='conv_3')(x)
        x = BatchNormalization(name='bn_3')(x)
        x = LeakyReLU(name='lrelu_3')(x)
    

        x = Conv2D(64, kernel_size=3, strides=1, padding='same', name='conv_4')(x)
        x = BatchNormalization(name='bn_4')(x)
        x = LeakyReLU(name='lrelu_4')(x)
    
        
        flatten = Flatten()(x)
        mean = Dense(2, name='mean')(flatten)
        log_variance = Dense(2, name='log_variance')(flatten)
        model = Model(inputs, (mean, log_variance), name="Encoder")
        return model
    
    def create_decoder(self, input_shape: tuple):
    
        inputs = Input(shape=input_shape, name='input_layer')
        x = Dense(3136, name='dense_1')(inputs)
        x = Reshape((7, 7, 64), name='reshape_layer')(x)
    
        # Block-1
        x = Conv2DTranspose(64, kernel_size=3, strides=1, padding='same',name='conv_transpose_1')(x)
        x = BatchNormalization(name='bn_1')(x)
        x = LeakyReLU(name='lrelu_1')(x)
    
        # Block-2
        x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', name='conv_transpose_2')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = LeakyReLU(name='lrelu_2')(x)
        
        # Block-3
        x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', name='conv_transpose_3')(x)
        x = BatchNormalization(name='bn_3')(x)
        x = LeakyReLU(name='lrelu_3')(x)
        
        # Block-4
        outputs = Conv2DTranspose(1, 3, strides=1, padding='same', activation='sigmoid', name='conv_transpose_4')(x)
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