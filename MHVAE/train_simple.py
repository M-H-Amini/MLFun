from tensorflow.keras.datasets.mnist import load_data
from mh_vae import MHVAE

##  Load MNIST dataset...
(X, _), (_, _) = load_data()
X = X.reshape(-1, 784) / 255.
##  Create and train VAE...
vae = MHVAE(input_dim=(784,), latent_dim=20, regularization_const=10000)
vae.compile(optimizer='adam')
vae.fit(X, epochs=10, batch_size=64)
vae.save_weights('vae_weights_20d.h5')