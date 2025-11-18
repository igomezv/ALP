#!/usr/bin/env python

"""
Astro Layer Perceptron Networks Module - Custom TensorFlow Layers
-------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
-------------------------------------------------------
Collection of custom TensorFlow/Keras layers and helper functions
used in Astro Layer Perceptron for neural network regularization, uncertainty
quantification (Monte Carlo dropout), and variational autoencoders.
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


class KLDivergenceLayer(K.layers.Layer):
    """
    Custom layer that computes and adds the KL divergence loss term.

    Used in Variational Autoencoders (VAEs) to regularize the latent space.

    Parameters
    ----------
    beta : float, optional
        Scaling factor for the KL divergence loss (default is 1.0).
    """

    def __init__(self, beta=1.0, **kwargs):
        super(KLDivergenceLayer, self).__init__(**kwargs)
        self.beta = beta  # scaling factor for KL loss

    def call(self, inputs):
        """
        Compute the KL divergence and add it as a model loss.

        Parameters
        ----------
        inputs : list of tf.Tensor
            [z_mean, z_log_sigma] from the encoder.

        Returns
        -------
        tf.Tensor
            The mean latent vector (z_mean) for downstream processing.
        """
        z_mean, z_log_sigma = inputs

        kl = 1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        kl = tf.reduce_sum(kl, axis=-1)
        kl_loss = -0.5 * self.beta * tf.reduce_mean(kl)

        self.add_loss(kl_loss)
        return z_mean  
    

class ReconstructionLossLayer(K.layers.Layer):
    """
    Custom layer to compute and add a reconstruction (MSE) loss term.

    Typically used in autoencoders or VAEs to penalize the difference
    between input and reconstructed output.
    """

    def call(self, inputs):
        """
        Compute the mean squared reconstruction loss.

        Parameters
        ----------
        inputs : list of tf.Tensor
            [x_true, x_pred] - ground truth and reconstructed tensors.

        Returns
        -------
        tf.Tensor
            The predicted tensor (x_pred), passed through unchanged.
        """
        x_true, x_pred = inputs
        loss = tf.reduce_mean(tf.square(x_true - x_pred), axis=-1)
        self.add_loss(tf.reduce_mean(loss))
        return x_pred  # passthrough
    

class Resize1DTensorLayer(K.layers.Layer):
    """
    Custom layer that resizes a 1D tensor using bilinear interpolation.

    Useful for upsampling feature sequences in 1D convolutional models.
    """

    def __init__(self, target_len, **kwargs):
        super().__init__(**kwargs)
        self.target_len = target_len

    def call(self, inputs):
        """
        Resize the temporal dimension of a 1D tensor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch, time, channels).

        Returns
        -------
        tf.Tensor
            Resized tensor with temporal dimension = target_len.
        """
        t = tf.expand_dims(inputs, axis=1) 
        t = tf.image.resize(t, size=[1, self.target_len], method='bilinear') 
        t = tf.squeeze(t, axis=1)  
        return t

    def get_config(self):
        """Return configuration for serialization."""
        config = super().get_config()
        config.update({'target_len': self.target_len})
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


def sampling(args):
    """
    Reparameterization trick for VAEs.

    Samples from a normal distribution using the mean and log-variance.

    Parameters
    ----------
    args : list of tf.Tensor
        [z_mean, z_log_sigma] from encoder output.

    Returns
    -------
    tf.Tensor
        Sampled latent vector.
    """
    z_mean, z_log_sigma = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))
    return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon