#!/usr/bin/env python
"""Hyperparameter Optimization for Pantheon SNIa using Optuna NSGA-II"""

import os
import numpy as np
import tensorflow as tf
import optuna
from optuna.samplers import NSGAIISampler

from alp.data.datasets import load_pantheon_data, preprocess_pantheon_data
from alp.networks.mlp import MLP
from alp.utils.gpu_config import setup_tensorflow_for_training
from alp.utils.logger_config import logger


def optimize_hyperparameters(z_train, y_train, z_test, y_test, n_trials=50):
    """Optimize hyperparameters using Optuna with NSGA-II genetic algorithm."""

    def objective(trial):
        """Objective function for Optuna optimization."""
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 128, step=16)
        dropout = trial.suggest_float("dropout", 0.05, 0.3, step=0.05)
        
        num_layers = trial.suggest_int("num_layers", 2, 4)
        layer_width = trial.suggest_int("layer_width", 100, 300, step=50)
        deep = [layer_width] * num_layers

        try:
            model = MLP(
                n_inputs=1,
                deep=deep,
                dropout=dropout,
                mcdropout=True,
                n_outputs=2,
            )
            keras_model = model.model_tf()
            keras_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse"
            )

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

            val_loss = history.history["val_loss"][-1]
            n_params = keras_model.count_params() / 1000

            return val_loss, n_params

        except Exception as e:
            logger.warning(f"Trial failed: {str(e)}")
            return float("inf"), float("inf")

    logger.info("Starting hyperparameter optimization with Optuna NSGA-II...")
    sampler = NSGAIISampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        directions=["minimize", "minimize"],
        study_name="pantheon_hyperparameters",
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_trial = study.best_trials[0]

    logger.info(f"\nOptimization completed!")
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best validation loss: {best_trial.values[0]:.4f}")
    logger.info(f"Best model complexity: {best_trial.values[1]:.2f}k parameters")
    logger.info(f"Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

    num_layers = best_trial.params["num_layers"]
    layer_width = best_trial.params["layer_width"]
    best_trial.params["deep"] = [layer_width] * num_layers
    
    return best_trial.params


def main():
    """Main hyperparameter optimization function for Pantheon."""
    logger.info("Starting Pantheon hyperparameter optimization")
    
    setup_tensorflow_for_training(seed=42, force_cpu=True)
    
    logger.info("Loading Pantheon data...")
    z_data, mb_data, dmb_data = load_pantheon_data("data/pantheon_lcparam_full_long_zhel.txt")
    z_train, z_test, y_train, y_test, scaler = preprocess_pantheon_data(z_data, mb_data, dmb_data)
    
    logger.info(f"Data loaded: {len(z_train)} training samples, {len(z_test)} test samples")
    logger.info(f"Z range: {np.min(z_data):.4f} - {np.max(z_data):.4f}")
    logger.info(f"mb range: {np.min(mb_data):.4f} - {np.max(mb_data):.4f}")
    logger.info(f"dmb range: {np.min(dmb_data):.4f} - {np.max(dmb_data):.4f}")
    logger.info(f"Training mb range: {np.min(y_train[:, 0]):.4f} - {np.max(y_train[:, 0]):.4f}")
    
    logger.info("\n" + "=" * 70)
    logger.info("Hyperparameter Optimization")
    logger.info("=" * 70)
    best_params = optimize_hyperparameters(z_train, y_train, z_test, y_test, n_trials=50)
    logger.info("=" * 70)
    
    os.makedirs("experiments/pantheon/outputs", exist_ok=True)
    import json
    
    params_to_save = best_params.copy()
    params_to_save["deep"] = params_to_save["deep"]
    
    with open("experiments/pantheon/outputs/best_hyperparameters.json", "w") as f:
        json.dump(params_to_save, f, indent=2)
    
    logger.info(f"Best hyperparameters saved to experiments/pantheon/outputs/best_hyperparameters.json")
    logger.info("Pantheon hyperparameter optimization completed successfully!")


if __name__ == "__main__":
    main()
