#!/usr/bin/env python

"""
Astro Layer Perceptron Networks Module - Base Classes
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
This module defines the foundational classes for all neural
network architectures in Astro Layer Perceptron, including supervised
models (e.g., CNNs, MLPs) and variational autoencoders (VAEs).
It provides shared functionality for dropout handling,
Monte Carlo uncertainty estimation, and model loading.
"""

import numpy as np
import tensorflow as tf
from .net_blocks import (
    MCDropout,
    KLDivergenceLayer,
    ReconstructionLossLayer,
    Resize1DTensorLayer,
    L1LatentRegularization,
    sampling,
)
from ..utils.logger_config import logger


class SuperVAE:
    """
    Base class for Variational Autoencoders (VAEs).

    Provides shared functionality for VAE subclasses, including
    Monte Carlo dropout prediction and model loading with
    Astro Layer Perceptron’s custom TensorFlow layers.

    Parameters
    ----------
    latent_dim : int
        Dimensionality of the latent representation.
    dropout : float, optional
        Dropout rate applied to network layers (default is 0.2).
    actfn : str, optional
        Activation function used in layers (default is 'tanh').
    mcdropout : bool, optional
        Enables Monte Carlo Dropout during inference (default is True).
    """

    def __init__(self, latent_dim, dropout=0.2, actfn='tanh', mcdropout=True):
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.mcdropout = mcdropout
        self.actfn = actfn

        logger.info(f"TensorFlow Version: {tf.__version__}")
        logger.info(f"Num GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")

    def mcdo_predict(self, testset, encoder, decoder, mc_dropout_num=50):
        """
        Perform Monte Carlo Dropout predictions for a VAE.

        Runs multiple stochastic forward passes through the encoder
        and decoder to estimate predictive uncertainty.

        Parameters
        ----------
        testset : array-like
            Input data to perform stochastic predictions on.
        encoder : tf.keras.Model
            Encoder part of the VAE.
        decoder : tf.keras.Model
            Decoder part of the VAE.
        mc_dropout_num : int, optional
            Number of Monte Carlo passes to perform (default is 50).

        Returns
        -------
        dict
            Dictionary containing means and standard deviations of
            encoded and decoded predictions:
            {
                'mean_encoder': np.ndarray,
                'std_encoder': np.ndarray,
                'mean': np.ndarray,
                'std': np.ndarray
            }
        """
        predictions_enc = np.array([encoder(testset, training=True)[0] for _ in range(mc_dropout_num)])
        predictions_dec = np.array([decoder(predictions_enc[i], training=True) for i in range(mc_dropout_num)])

        return {
            'mean_encoder': np.mean(predictions_enc, axis=0),
            'std_encoder': np.std(predictions_enc, axis=0, ddof=1),
            'mean': np.mean(predictions_dec, axis=0),
            'std': np.std(predictions_dec, axis=0, ddof=1)
        }

    def load_model(self, model_name):
        """
        Load a trained VAE model with Astro Layer Perceptron custom layers.

        Parameters
        ----------
        model_name : str
            Path to the saved model file.

        Returns
        -------
        tf.keras.Model
            Loaded TensorFlow model.
        """
        custom_objects = {
            'MCDropout': MCDropout,
            'sampling': sampling,
            'KLDivergenceLayer': KLDivergenceLayer,
            'ReconstructionLossLayer': ReconstructionLossLayer,
            'Resize1DTensorLayer': Resize1DTensorLayer,
            'L1LatentRegularization': L1LatentRegularization,
        }
        return tf.keras.models.load_model(model_name, custom_objects=custom_objects)

    def model_tf(self):
        """
        Abstract method to build the TensorFlow model.

        Subclasses must implement this method to define
        the encoder, decoder, and full VAE architecture.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class SupervisedNET:
    """
    Base class for supervised neural networks (MLPs, CNNs, etc.).

    Provides shared initialization, dropout management, and
    Monte Carlo Dropout prediction methods for regression or
    classification networks.

    Parameters
    ----------
    deep : list of int, optional
        Defines the number of neurons in each dense layer (default [100, 100, 100]).
    actfn : str, optional
        Activation function for layers (default is 'relu').
    dropout : float, optional
        Dropout rate applied to layers (default is 0.2).
    mcdropout : bool, optional
        Enables Monte Carlo Dropout for uncertainty estimation (default is True).
    """

    def __init__(self, deep=None, actfn='relu', dropout=0.2, mcdropout=True):
        self.dropout = dropout
        self.mcdropout = mcdropout
        self.actfn = actfn
        self.deep = deep if deep is not None else [100, 100, 100]

        logger.info(f"TensorFlow Version: {tf.__version__}")

    def mcdo_predict(self, testset, model, mc_dropout_num=50):
        """
        Perform Monte Carlo Dropout predictions for supervised models.

        Parameters
        ----------
        testset : array-like
            Input dataset for prediction.
        model : tf.keras.Model
            Trained Keras model with dropout layers.
        mc_dropout_num : int, optional
            Number of stochastic forward passes (default is 50).

        Returns
        -------
        dict
            Dictionary containing mean and standard deviation of predictions:
            {
                'mean': np.ndarray,
                'std': np.ndarray
            }
        """
        predictions = np.array([model(testset, training=True) for _ in range(mc_dropout_num)])
        return {
            'mean': np.mean(predictions, axis=0),
            'std': np.std(predictions, axis=0, ddof=1),
        }

    def load_model(self, model_name):
        """
        Load a trained supervised model with custom Astro Layer Perceptron layers.

        Parameters
        ----------
        model_name : str
            Path to the saved model file.

        Returns
        -------
        tf.keras.Model
            Loaded TensorFlow model.
        """
        custom_objects = {'MCDropout': MCDropout}
        return tf.keras.models.load_model(model_name, custom_objects=custom_objects)

    def model_tf(self):
        """
        Abstract method to build the TensorFlow model.

        Subclasses must implement this method to define
        their specific supervised architecture.
        """
        raise NotImplementedError("Subclasses must implement this method.")