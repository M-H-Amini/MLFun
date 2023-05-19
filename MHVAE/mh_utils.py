from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

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
    mean = Dense(latent_dim, 'tanh')(x)
    log_sigma = Dense(latent_dim)(x)
    model = Model(inputs=[inp], outputs=[mean, log_sigma])
    return model

def mhLoss(mean, log_sigma, X, X_hat):
    term_tr = tf.reduce_mean(tf.reduce_sum(tf.math.exp(log_sigma), axis=1))
    term_sq = tf.reduce_mean(tf.reduce_sum(mean * mean, axis=1))
    term_dt = tf.reduce_mean(tf.reduce_sum(log_sigma, axis=1))
    loss_kl = 0.5 * (term_tr + term_sq - term_dt)
    loss_p = tf.reduce_mean(tf.reduce_sum(tf.losses.mse(X_hat, X), axis=1))
    loss = loss_kl + 200 * loss_p
    return loss