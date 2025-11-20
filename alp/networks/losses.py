#!/usr/bin/env python
"""
Loss functions for ALP networks.
"""

import tensorflow as tf

def heteroscedastic_gaussian_nll(y_true, y_pred):
    """
    Gaussian negative log-likelihood for heteroscedastic regression.

    y_true: shape (batch, 1)
    y_pred: shape (batch, 2) -> [y_pred, log_sigma2_pred]
    """
    y = y_pred[:, 0]
    log_sigma2 = y_pred[:, 1]

    # avoid numerical issues
    log_sigma2 = tf.clip_by_value(log_sigma2, -20.0, 5.0)
    sigma2 = tf.exp(log_sigma2)

    # NLL up to additive constant
    nll = 0.5 * (log_sigma2 + tf.square(y_true[:, 0] - y) / sigma2)
    return tf.reduce_mean(nll)

