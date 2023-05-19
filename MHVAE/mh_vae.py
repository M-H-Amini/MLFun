#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                            Title: Variational Autoencoder                           ##
##                                   Date: 2023/05/18                                  ##
##                                                                                     ##
#########################################################################################

##  Description: Easy-to-use Variational Autoencoder (VAE) implementation in TensorFlow 2.0



from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from mh_utils import buildP, buildQ
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MHVAE(Model):
    def __init__(self, input_dim, latent_dim=2, model_p=None, model_q=None, regularization_const=10000, train_visualize=False):
        super(MHVAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model_p = model_p if model_p else buildP(input_dim, latent_dim)
        self.model_q = model_q if model_q else buildQ(input_dim, latent_dim)
        self.regularization_const = regularization_const
        self.train_visualize = train_visualize
        if self.train_visualize:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(8, 4))
            self.fig.show()
        self.train_step_cnt = 0
        self.cache = []
        self(tf.zeros((1, *input_dim)))

    def updateTrainVisualize(self, X, X_hat, title='Training VAE'):
        """Updates training visualization. After each call of this function, the figure will be updated.

        Args:
            X (tf.Tensor): Batch of images from training set.
            X_hat (tf.Tensor): Reconstructed images from VAE.
            title (str, optional): Title of the figure. Defaults to 'Training VAE'.
        """
        img = np.zeros((4 * 28, 8 * 28), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                img[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = (X[i * 4 + j].numpy().reshape((28, 28)) * 255).astype(np.uint8)
                img[i * 28:(i + 1) * 28, (j + 4) * 28:(j + 5) * 28] = (X_hat[i * 4 + j].numpy().reshape((28, 28)) * 255).astype(np.uint8)
        self.ax.clear()
        self.ax.imshow(img, cmap='gray')
        self.ax.set_axis_off()
        self.fig.suptitle(title)
        self.fig.canvas.draw()
        plt.pause(0.01)
        self.fig.show()
        self.cache.append(img)
        
    def call(self, x):
        """Forward pass of VAE.

        Args:
            x (tf.Tensor): Batch of images.

        Returns:
            tf.Tensor: Reconstructed images.
        """
        mean, log_sigma = self.model_q(x)
        eps = tf.random.normal(shape=tf.shape(mean))
        z = mean + tf.exp(log_sigma * .5) * eps
        return self.model_p(z)
    
    def encode(self, x):
        """Encodes a batch of images to latent space.

        Args:
            x (tf.Tensor): Batch of images.

        Returns:
            tf.Tensor: Mean and log_sigma of latent space.
        """
        mean, log_sigma = self.model_q(x)
        return mean, log_sigma
    
    def decode(self, z):
        """Decodes a batch of latent space vectors to images.

        Args:
            z (tf.Tensor): Batch of latent space vectors.

        Returns:
            tf.Tensor: Reconstructed images.
        """
        return self.model_p(z)
    
    def sample(self, batch_size=64):
        """Samples a batch of images from latent space.

        Args:
            batch_size (int, optional): Batch size. Defaults to 64.

        Returns:
            tf.Tensor: Reconstructed images.
        """
        z = tf.random.normal((batch_size, self.latent_dim))
        return self.model_p(z)
    
    def train_step(self, data):
        """Training step of VAE.

        Args:
            data (tf.Tensor): Batch of images.

        Returns:
            tf.Tensor: Loss value.
        """
        self.train_step_cnt -=- 1
        X = data 
        with tf.GradientTape() as g:
            mean, log_sigma = self.model_q(X)
            term_tr = tf.reduce_mean(tf.reduce_sum(tf.math.exp(log_sigma), axis=1))
            term_sq = tf.reduce_mean(tf.reduce_sum(mean * mean, axis=1))
            term_dt = tf.reduce_mean(tf.reduce_sum(log_sigma, axis=1))
            loss_kl = 0.5 * (term_tr + term_sq - term_dt)
            eps = tf.random.normal(shape=tf.shape(mean))
            z = mean + eps * tf.exp(0.5 * log_sigma)
            X_hat = self.model_p(z)
            loss_p = tf.reduce_mean(tf.losses.mse(X_hat, X))
            loss = loss_kl + self.regularization_const * loss_p
        grads = g.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        if self.train_visualize and self.train_step_cnt % 250 == 0:
            self.updateTrainVisualize(X, X_hat, title=f'Training VAE - Step: {self.train_step_cnt}')
        return {'loss': loss}
    
    def generateGIF(self, filename='training.gif'):
        if not self.train_visualize:
            raise Exception('train_visualize is False, cannot generate GIF')
        import imageio
        imageio.mimsave(filename, self.cache, fps=5)
        log.debug(f'GIF saved to {filename}')


if __name__ == '__main__':
    ##  Load MNIST dataset...
    (X, _), (_, _) = load_data()
    X = X.reshape(-1, 784) / 255.
    ##  Create and train VAE...
    vae = MHVAE((784,), latent_dim=20, regularization_const=10000)
    vae.compile(optimizer='adam')
    vae.fit(X, epochs=10, batch_size=64)
    vae.save_weights('vae_weights_20d.h5')
    # vae.generateGIF()
