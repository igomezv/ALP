#!/usr/bin/env python

"""
Astro Layer Perceptron Networks Module - Custom TensorFlow Layers
-------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
-------------------------------------------------------
Collection of custom TensorFlow/Keras layers and helper functions
used in Astro Layer Perceptron for neural network regularization, uncertainty
quantification (Monte Carlo dropout).
"""

import tensorflow as tf
from tensorflow import keras as K
from ..utils.logger_config import logger  


class MCDropout(K.layers.Layer):
    """
    Monte Carlo Dropout layer.

    Enables dropout to remain active during inference, allowing stochastic
    forward passes for uncertainty estimation and Bayesian approximation.

    Parameters
    ----------
    rate : float
        Dropout rate (probability of dropping a unit).
    is_disabled : bool, optional
        If True, disables dropout completely (default is False).
    noise_shape : tuple, optional
        Shape of the dropout noise mask (default is None).
    name : str, optional
        Name of the layer.
    **kwargs : dict
        Additional arguments passed to the Keras Layer base class.
    """

    def __init__(
            self, rate: float, is_disabled: bool = False,
            noise_shape: tuple = None, name: str = None, **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.rate = rate
        self.is_disabled = is_disabled
        self.noise_shape = noise_shape
        logger.info(f"MCDropout initialized with rate={self.rate}, is_disabled={self.is_disabled}")

    def call(self, inputs: tf.Tensor, training: bool = None) -> tf.Tensor:
        """
        Apply dropout during both training and inference (if not disabled).

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor to apply dropout on.
        training : bool, optional
            Whether the layer is currently in training mode.

        Returns
        -------
        tf.Tensor
            Tensor after dropout or unmodified input if disabled.
        """
        if self.is_disabled:
            return inputs
        return tf.nn.dropout(inputs, rate=self.rate, noise_shape=self.noise_shape)

    def get_config(self) -> dict:
        """Returns the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            'rate': self.rate,
            'is_disabled': self.is_disabled,
            'noise_shape': self.noise_shape,
        })
        return config


class L1LatentRegularization(K.layers.Layer):
    """
    Applies L1 regularization directly to latent variables in VAEs.

    Parameters
    ----------
    l1_lambda : float, optional
        Regularization coefficient (default is 1e-3).
    """

    def __init__(self, l1_lambda=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.l1_lambda = l1_lambda

    def call(self, z):
        """
        Add L1 regularization loss term on the latent vector.

        Parameters
        ----------
        z : tf.Tensor
            Latent representation tensor.

        Returns
        -------
        tf.Tensor
            The same latent tensor, passed through unchanged.
        """
        self.add_loss(self.l1_lambda * tf.reduce_sum(tf.abs(z)))
        return z


def resize_1d_tensor(t, target_len):
    """
    Resize a 1D tensor along its temporal dimension.

    Parameters
    ----------
    t : tf.Tensor
        Input tensor of shape (batch, time, channels).
    target_len : int
        Target temporal length after resizing.

    Returns
    -------
    tf.Tensor
        Resized tensor with new temporal length.
    """
    t = tf.expand_dims(t, axis=1)  # → (batch, 1, time, channels)
    t = tf.image.resize(t, size=[1, target_len], method='bilinear')  # Valid 2D resize
    t = tf.squeeze(t, axis=1)  # → (batch, time, channels)
    return t