#########################################################################################
##                                                                                     ##
##                                Mohammad Hossein Amini                               ##
##                               Title: Sparse Autoencoder                             ##
##                                   Date: 2023/05/20                                  ##
##                                                                                     ##
#########################################################################################

##  Description: Easy-to-use Sparse Autoencoder (VAE) implementation in TensorFlow 2.0



from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, Reshape, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import logging as log

log.basicConfig(level=log.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MHAE(Model):
    def __init__(self, input_dim, latent_dim=2, model_p=None, model_q=None, regularization_const=10000, train_visualize=False):
        super(MHAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.model_p = model_p if model_p else buildP(input_dim[0], latent_dim)
        self.model_q = model_q if model_q else buildQ(input_dim[0], latent_dim)
        self.regularization_const = regularization_const
        self.train_visualize = train_visualize
        if self.train_visualize:
            self.fig, self.ax = plt.subplots(figsize=(8, 4))
            self.fig.show()
        self.train_step_cnt = 0
        self.cache = []
        self(tf.zeros((1, *input_dim)))

    def updateTrainVisualize(self, X, X_hat, title='Training AE'):
        """Updates training visualization. After each call of this function, the figure will be updated.

        Args:
            X (tf.Tensor): Batch of images from training set.
            X_hat (tf.Tensor): Reconstructed images from VAE.
            title (str, optional): Title of the figure. Defaults to 'Training VAE'.
        """
        if len(X) < 16:
            return
        h, w = 28, 28
        img = np.zeros((4 * h, 8 * w), dtype=np.uint8)
        for i in range(4):
            for j in range(4):
                img[i * h:(i + 1) * h, j * w:(j + 1) * w] = (X[i * 4 + j].numpy().reshape((h, w)) * 255).astype(np.uint8)
                img[i * h:(i + 1) * h, (j + 4) * w:(j + 5) * w] = (X_hat[i * 4 + j].numpy().reshape((h, w)) * 255).astype(np.uint8)
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_axis_off()
        self.ax.set_title(title)
        plt.pause(0.01)
        self.fig.canvas.draw()
        plt.show(block=False)
        self.cache.append(img)
        
    def call(self, x):
        """Forward pass of VAE.

        Args:
            x (tf.Tensor): Batch of images.

        Returns:
            tf.Tensor: Reconstructed images.
        """
        z = self.model_q(x)
        return self.model_p(z)
    
    def encode(self, x):
        """Encodes a batch of images to latent space.

        Args:
            x (tf.Tensor): Batch of images.

        Returns:
            tf.Tensor: Mean and log_sigma of latent space.
        """
        return self.model_q(x)
    
    def decode(self, z):
        """Decodes a batch of latent space vectors to images.

        Args:
            z (tf.Tensor): Batch of latent space vectors.

        Returns:
            tf.Tensor: Reconstructed images.
        """
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
            z = self.model_q(X)
            X_hat = self.model_p(z)
            loss = tf.reduce_mean(tf.square(X - X_hat))
        grads = g.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        if self.train_visualize and self.train_step_cnt % 50 == 0:
            self.updateTrainVisualize(X, X_hat, title=f'Training SAE - Step: {self.train_step_cnt}')
        return {'loss': loss}
    
    def generateGIF(self, filename='sae_training.gif'):
        if not self.train_visualize:
            raise Exception('train_visualize is False, cannot generate GIF')
        import imageio
        imageio.mimsave(filename, self.cache, fps=5)
        log.debug(f'GIF saved to {filename}')

def buildP(input_shape=784, latent_dim=2):
    inp = Input((latent_dim,))
    x = Dense(40, 'sigmoid')(inp)
    x = Dense(100, 'sigmoid')(x)
    x = Dense(input_shape, 'sigmoid')(x)
    model = Model(inputs=[inp], outputs=[x])
    return model

def buildQ(input_shape=784, latent_dim=2):
    inp = Input((input_shape,))
    x = Dense(100, 'sigmoid')(inp)
    x = Dense(40, 'sigmoid')(x)
    z = Dense(latent_dim, 'tanh', activity_regularizer=tf.keras.regularizers.l1(0.01))(x)
    model = Model(inputs=[inp], outputs=[z])
    return model

if __name__ == '__main__':
    ##  Load MNIST dataset...
    (X, _), (_, _) = load_data()
    X = X.reshape((-1, 784)) / 255.0
    ##  Create and train VAE...
    sae = MHAE((784,), latent_dim=20, train_visualize=True)
    sae.compile(optimizer='adam', loss='mse', run_eagerly=True)
    sae.fit(X, epochs=10, batch_size=64)
    sae.save_weights('sae_weights_20d.h5')
    sae.generateGIF()
