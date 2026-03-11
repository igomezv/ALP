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
    
    # Optimized hyperparameters from Optuna NSGA-II search
    LEARNING_RATE = 0.00582938454299474
    BATCH_SIZE = 16
    DROPOUT = 0.1
    DEEP_LAYERS = [150, 150]
    EPOCHS = 1000
    PATIENCE = 500
    MC_DROPOUT_SAMPLES = 100
    
    # Create and compile model
    logger.info("Creating ALP dual-output MLP...")
    model = MLP(n_inputs=1, deep=DEEP_LAYERS, dropout=DROPOUT, mcdropout=True, n_outputs=2)
    keras_model = model.model_tf()
    keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), loss="mse")
    
    # Train model
    logger.info("Training model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
    )
    history = keras_model.fit(
        z_train,
        y_train,
        validation_data=(z_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=2,
        callbacks=[early_stopping],
    )
    
    # Uncertainty quantification
    logger.info("Performing MC Dropout uncertainty quantification...")
    uq = UncertaintyQuantifier(n_samples=MC_DROPOUT_SAMPLES)
    z_test_range = np.linspace(np.min(z_data), np.max(z_data), 500)
    results = uq.mc_dropout_prediction(keras_model, z_test_range, scaler)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    keras_model.save("models/alp_lsst_model.h5")
    logger.info("Model saved to models/alp_lsst_model.h5")
    
    logger.info("LSST training completed successfully!")


if __name__ == "__main__":
    main()
