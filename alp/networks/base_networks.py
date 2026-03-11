#!/usr/bin/env python

"""
Astro Layer Perceptron Networks Module - Base Classes
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
This module defines the foundational class for all neural
network architectures in Astro Layer Perceptron.
It provides shared functionality for dropout handling,
Monte Carlo uncertainty estimation, and model loading.
"""

import numpy as np
import tensorflow as tf
from .net_blocks import MCDropout
from ..utils.logger_config import logger
from ..utils.gpu_config import get_device_name, is_gpu_available


class SupervisedNET:
    """
    Base class for supervised neural networks (MLPs, CNNs, etc.).

    Provides shared initialization, dropout management, and
    Monte Carlo Dropout prediction methods for regression or
    classification networks. Automatically handles GPU/CPU device
    management with fallback to CPU if GPU is unavailable.

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

    Notes
    -----
    Device Management:
    - Models automatically run on available GPU if configured via setup_tensorflow_for_training()
    - Falls back to CPU automatically if GPU is unavailable or encounters errors
    - Use with create_safe_dataset() for GPU-safe data loading
    """

    def __init__(self, deep=None, actfn="relu", dropout=0.2, mcdropout=True):
        self.dropout = dropout
        self.mcdropout = mcdropout
        self.actfn = actfn
        self.deep = deep if deep is not None else [100, 100, 100]

        logger.info(f"TensorFlow Version: {tf.__version__}")
        if is_gpu_available():
            logger.info(f"GPU available, models will train on: {get_device_name()}")
        else:
            logger.info("Training on CPU")

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
            "mean": np.mean(predictions, axis=0),
            "std": np.std(predictions, axis=0, ddof=1),
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
        custom_objects = {"MCDropout": MCDropout, "mse": tf.keras.losses.mse}
        return tf.keras.models.load_model(model_name, custom_objects=custom_objects)

    def model_tf(self):
        """
        Abstract method to build the TensorFlow model.

        Subclasses must implement this method to define
        their specific supervised architecture.
        """
        raise NotImplementedError("Subclasses must implement this method.")
