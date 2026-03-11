#!/usr/bin/env python
"""Hyperparameter Optimization for Cosmic Chronometers (CC) using Optuna NSGA-II"""

import os
import numpy as np
import tensorflow as tf
import optuna
from optuna.samplers import NSGAIISampler

from alp.data.datasets import load_hz31_data, preprocess_cc_data
from alp.networks.mlp import MLP
from alp.utils.gpu_config import setup_tensorflow_for_training
from alp.utils.logger_config import logger


def optimize_hyperparameters(z_train, y_train, z_test, y_test, n_trials=50):
    """Optimize hyperparameters using Optuna with NSGA-II genetic algorithm.
    
    Parameters
    ----------
    z_train : np.ndarray
        Training redshift values
    y_train : np.ndarray
        Training target values
    z_test : np.ndarray
        Test redshift values
    y_test : np.ndarray
        Test target values
    n_trials : int
        Number of optimization trials (default: 50)
    
    Returns
    -------
    dict
        Optimal hyperparameters: learning_rate, batch_size, dropout_rate, layer_width
    """

    def objective(trial):
        """Objective function for Optuna optimization.
        
        Objectives:
        1. Minimize validation loss
        2. Minimize model complexity (parameters)
        """
        # Hyperparameters to optimize
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_int("batch_size", 8, 32, step=8)
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.3, step=0.05)
        layer_width = trial.suggest_int("layer_width", 64, 256, step=32)

        try:
            # Create model with suggested hyperparameters
            model = MLP(
                n_inputs=1,
                deep=[layer_width, layer_width, layer_width, layer_width],
                dropout=dropout_rate,
                mcdropout=True,
                n_outputs=2,
            )
            keras_model = model.model_tf()

            # Compile
            keras_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
            )

            # Train with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=200, restore_best_weights=True, verbose=0
            )

            history = keras_model.fit(
                z_train,
                y_train,
                validation_data=(z_test, y_test),
                epochs=200,
                batch_size=batch_size,
                verbose=0,
                callbacks=[early_stopping],
            )

            # Objectives: minimize validation loss and model complexity
            val_loss = history.history["val_loss"][-1]
            n_params = keras_model.count_params() / 1000  # In thousands

            return val_loss, n_params

        except Exception as e:
            logger.warning(f"Trial failed: {str(e)}")
            return float("inf"), float("inf")

    # Create study with NSGA-II sampler (multi-objective optimization)
    logger.info("Starting hyperparameter optimization with Optuna NSGA-II...")
    sampler = NSGAIISampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        directions=["minimize", "minimize"],  # Minimize loss and complexity
        study_name="cc_hyperparameters",
    )

    # Optimize
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # Get best trial (based on validation loss as primary objective)
    best_trial = study.best_trials[0]

    logger.info(f"\nOptimization completed!")
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best validation loss: {best_trial.values[0]:.4f}")
    logger.info(f"Best model complexity: {best_trial.values[1]:.2f}k parameters")
    logger.info(f"Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

    return best_trial.params


def main():
    """Main hyperparameter optimization function for CC."""
    logger.info("Starting CC (Cosmic Chronometers) hyperparameter optimization")
    
    # Setup TensorFlow
    setup_tensorflow_for_training(seed=42, force_cpu=True)
    
    # Load and preprocess data
    logger.info("Loading CC H(z) data...")
    here = os.path.dirname(__file__)
    data_file = os.path.join(here, "..", "..", "data", "Hz31.txt")
    z_data, hz_data, error_data = load_hz31_data(data_file)
    z_train, z_test, y_train, y_test, scaler = preprocess_cc_data(z_data, hz_data, error_data)
    
    logger.info(f"Data loaded: {len(z_train)} training samples, {len(z_test)} test samples")
    logger.info(f"Z range: {np.min(z_data):.4f} - {np.max(z_data):.4f}")
    logger.info(f"H(z) range: {np.min(hz_data):.4f} - {np.max(hz_data):.4f}")
    logger.info(f"Error range: {np.min(error_data):.4f} - {np.max(error_data):.4f}")
    logger.info(f"Training H(z) range: {np.min(y_train[:, 0]):.4f} - {np.max(y_train[:, 0]):.4f}")
    
    # Optimize hyperparameters
    logger.info("\n" + "=" * 70)
    logger.info("Hyperparameter Optimization")
    logger.info("=" * 70)
    best_params = optimize_hyperparameters(z_train, y_train, z_test, y_test, n_trials=50)
    logger.info("=" * 70)
    
    # Save results
    os.makedirs("experiments/cc/outputs", exist_ok=True)
    import json
    
    with open("experiments/cc/outputs/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)
    
    logger.info(f"Best hyperparameters saved to experiments/cc/outputs/best_hyperparameters.json")
    logger.info("CC hyperparameter optimization completed successfully!")


if __name__ == "__main__":
    main()
