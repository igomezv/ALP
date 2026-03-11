#!/usr/bin/env python
"""Integrated LSST Training Script using ALP Framework"""

import os
import numpy as np
import tensorflow as tf

from alp.data.datasets import load_lsst_data, preprocess_lsst_data
from alp.data.data_reading import force_x_range_in_training
from alp.networks.mlp import MLP
from alp.networks.uncertainty import UncertaintyQuantifier
from alp.utils.gpu_config import setup_tensorflow_for_training
from alp.utils.logger_config import logger


def main():
    """Train LSST dual-output regression model using ALP framework."""
    
    # Setup TensorFlow
    setup_tensorflow_for_training(seed=42, force_cpu=True)
    
    # Load and preprocess data
    z_data, mu_data, error_data = load_lsst_data()
    z_train, z_test, y_train, y_test, scaler = preprocess_lsst_data(z_data, mu_data, error_data)
    z_train, y_train, _ = force_x_range_in_training(
        z_train, y_train, np.concatenate([z_train, z_test]), np.concatenate([y_train, y_test])
    )
    
    # Create and compile model
    logger.info("Creating ALP dual-output MLP...")
    model = MLP(n_inputs=1, deep=[200, 200, 200, 200], dropout=0.1, mcdropout=True, n_outputs=2)
    keras_model = model.model_tf()
    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
    
    # Train model
    logger.info("Training model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=500, restore_best_weights=True, verbose=1
    )
    history = keras_model.fit(
        z_train,
        y_train,
        validation_data=(z_test, y_test),
        epochs=1000,
        batch_size=16,
        verbose=2,
        callbacks=[early_stopping],
    )
    
    # Uncertainty quantification
    logger.info("Performing MC Dropout uncertainty quantification...")
    uq = UncertaintyQuantifier(n_samples=100)
    z_test_range = np.linspace(np.min(z_data), np.max(z_data), 500)
    results = uq.mc_dropout_prediction(keras_model, z_test_range, scaler)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    keras_model.save("models/alp_lsst_model.h5")
    logger.info("Model saved to models/alp_lsst_model.h5")
    
    logger.info("LSST training completed successfully!")


if __name__ == "__main__":
    main()
