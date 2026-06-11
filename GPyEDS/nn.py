import numpy as np
from pathlib import Path
import tensorflow as tf


def create_nn_AE(input_dim, latent_dim = 2, hidden = [10], activation = "relu"):
    """Generator function for neural network autoencoder architecture. 

    Args:
        input_dim (int): Input dimensions for model.
        latent_dim (int, optional): Latent space dimensions/bottleneck size. Defaults to 2.
        hidden (list, optional): Dimensions for hidden layers - note model will be symmetric. Defaults to [10].
        activation (str, optional): Activation functions to use. Defaults to "relu".

    Returns:
        model (TF model): Autoencoder model.
    """

    enc_list = []
    for i in range(len(hidden)):
        enc_list.append(tf.keras.layers.Dense(hidden[i], activation=activation))
        enc_list.append(tf.keras.layers.LayerNormalization())
        enc_list.append(tf.keras.layers.LeakyReLU(0.02))

    dec_list = []
    for i in range(len(hidden)):
        dec_list.append(tf.keras.layers.Dense(hidden[::-1][i], activation = activation))
        dec_list.append(tf.keras.layers.LayerNormalization())
        dec_list.append(tf.keras.layers.LeakyReLU(0.02))

    dec_list.append(tf.keras.layers.Dense(input_dim))

    # Keras 3 functional API: connect explicit Input through the sub-models
    # instead of reading Sequential.input / .output (Keras-2-only).
    inputs = tf.keras.Input(shape=(input_dim,))
    encoder = tf.keras.Sequential(enc_list)
    latent = tf.keras.Sequential([tf.keras.layers.Dense(latent_dim)])
    decoder = tf.keras.Sequential(dec_list)

    model = tf.keras.Model(inputs=inputs, outputs=decoder(latent(encoder(inputs))))

    return model, (encoder, latent, decoder)

