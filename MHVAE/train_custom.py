from tensorflow.keras.datasets.mnist import load_data
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from mh_vae import MHVAE

##  Loading data...
(X, y), (_, _) = load_data()
X = X.reshape((*X.shape, 1)) / 255.

##  Building models...
latent_dim = 2

###  Q model (encoder)...
inp = tf.keras.layers.Input(shape=(28, 28, 1))
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation='relu')(inp)
x = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation='relu')(x)
x = tf.keras.layers.Flatten()(x)
mean = tf.keras.layers.Dense(latent_dim, activation='sigmoid')(x)
log_sigma = tf.keras.layers.Dense(latent_dim)(x)
model_q = tf.keras.models.Model(inputs=inp, outputs=[mean, log_sigma])
model_q.summary()

###  P model (decoder)...
inp = tf.keras.layers.Input(shape=(latent_dim,))
x = tf.keras.layers.Dense(7 * 7 * 32, activation='relu')(inp)
x = tf.keras.layers.Reshape((7, 7, 32))(x)
x = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation='relu')(x)
x = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same', activation='sigmoid')(x)
model_p = tf.keras.models.Model(inputs=inp, outputs=x)
model_p.summary()

###  MHVAE model...
model = MHVAE(input_dim=(28, 28, 1), latent_dim=latent_dim, model_p=model_p, model_q=model_q, regularization_const=10000, train_visualize=True)
model.compile(optimizer='adam', run_eagerly=True)
model.fit(X, epochs=15, batch_size=64)
model.generateGIF('mh_cvae.gif')
model.save_weights('mh_cvae_weights.h5')