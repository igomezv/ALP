#!/usr/bin/env python

"""
Astro Layer Perceptron Networks Module - Multilayer Perceptron (MLP)
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
Feedforward Multilayer Perceptron (MLP) architectures for
supervised regression or classification tasks using shell-
based or spectral inputs.
"""

from tensorflow import keras as K
from .base_networks import SupervisedNET
from .net_blocks import MCDropout
from ..utils.logger_config import logger 


class MLP(SupervisedNET):
    """
    Feedforward Multilayer Perceptron (MLP) for supervised tasks.

    A flexible fully-connected neural network suitable for regression
    or classification using shell representations, spectra, or derived
    astrophysical data.

    Parameters
    ----------
    n_inputs : int
        Number of input features (e.g., flattened shell size or spectral bins).
    n_outputs : int
        Number of outputs (e.g., RV, BIS, or other scalar targets).
    **kwargs : dict
        Additional arguments for the parent `SupervisedNET` class, such as
        dropout rate, activation function, and architecture depth.
    """

    def __init__(self, n_inputs, n_outputs, **kwargs):
        super().__init__(**kwargs)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def model_tf(self):
        logger.info("Building FFNN model...")

        inputs = K.layers.Input(shape=(self.n_inputs,), name='mlp_input')
        x = inputs

        for neurons in self.deep:
            x = K.layers.Dense(neurons, activation=self.actfn)(x)
            if self.mcdropout:
                x = MCDropout(self.dropout)(x)

        outputs = K.layers.Dense(self.n_outputs, activation='linear')(x)
        model = K.Model(inputs, outputs, name='mlp')
        logger.info(model.summary())
        return model