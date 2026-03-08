#!/usr/bin/env python
"""Hyperparameter Optimization for LSST Model using Optuna

This script uses Optuna to automatically find the best hyperparameters
for the LSST dual-output regression model.
"""

import os
import json
import optuna
import tensorflow as tf
import numpy as np

from alp.networks.mlp import MLP
from alp.data.datasets import load_lsst_data, preprocess_lsst_data
from alp.utils.logger_config import logger


def objective(trial):
    """Objective function for Optuna optimization.

    Args:
        trial: Optuna trial object

    Returns:
        float: Best validation loss achieved
    """

    # Hyperparameters to optimize
    deep = trial.suggest_categorical(
        "deep",
        [
            [100, 100, 100],
            [200, 200, 200, 200],
            [300, 300, 300],
            [128, 256, 128, 64],
            [64, 128, 256, 128, 64],
        ],
    )
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    logger.info(
        f"Trial {trial.number}: deep={deep}, dropout={dropout:.3f}, lr={lr:.6f}, batch_size={batch_size}"
    )

    try:
        # Load and preprocess data
        z_data, mu_data, error_data = load_lsst_data()
        z_train, z_test, y_train, y_test, scaler = preprocess_lsst_data(
            z_data, mu_data, error_data, train_split=0.8, random_state=42
        )

        # Create model
        model = MLP(n_inputs=1, deep=deep, dropout=dropout, mcdropout=True, n_outputs=2)
        keras_model = model.model_tf()

        # Compile and train
        keras_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=100, restore_best_weights=True, verbose=0
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=50, min_lr=1e-7, verbose=0
        )

        history = keras_model.fit(
            z_train,
            y_train,
            validation_data=(z_test, y_test),
            epochs=1000,  # Reduced for optimization speed
            batch_size=batch_size,
            verbose=0,
            callbacks=[early_stopping, reduce_lr],
        )

        # Return best validation loss
        val_loss = min(history.history["val_loss"])
        logger.info(f"Trial {trial.number}: val_loss={val_loss:.4f}")

        # Save trial results
        trial_results = {
            "trial_number": trial.number,
            "hyperparameters": trial.params,
            "val_loss": float(val_loss),
            "training_history": {
                "loss": [float(x) for x in history.history["loss"]],
                "val_loss": [float(x) for x in history.history["val_loss"]],
            },
        }

        return val_loss

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float("inf")


def optimize_hyperparameters(n_trials=50, timeout=3600):
    """Run Optuna hyperparameter optimization.

    Args:
        n_trials (int): Number of optimization trials
        timeout (int): Maximum time in seconds

    Returns:
        optuna.Study: Completed study object
    """

    # Create study with advanced sampling and pruning
    study = optuna.create_study(
        study_name="lsst_alp_optimization",
        direction="minimize",
        sampler=optuna.samplers.NSGAIISampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    logger.info(f"Starting hyperparameter optimization with {n_trials} trials, {timeout}s timeout")

    # Optimize with progress monitoring
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

    # Print best results
    logger.info("Optimization completed!")
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value (val_loss): {trial.value:.6f}")
    logger.info(f"  Best hyperparameters:")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save best parameters and study results
    os.makedirs("experiments/lsst/outputs", exist_ok=True)

    # Save best hyperparameters
    best_params = {
        "best_val_loss": float(trial.value),
        "best_hyperparameters": trial.params,
        "trial_number": trial.number,
    }

    with open("experiments/lsst/outputs/best_hyperparameters.json", "w") as f:
        json.dump(best_params, f, indent=2)

    # Save complete study
    study_data = {
        "best_trial": {
            "number": trial.number,
            "value": float(trial.value),
            "params": trial.params,
        },
        "n_trials": len(study.trials),
        "study_name": study.study_name,
    }

    with open("experiments/lsst/outputs/optuna_study_summary.json", "w") as f:
        json.dump(study_data, f, indent=2)

    logger.info("Results saved to experiments/lsst/outputs/")

    return study


def plot_optimization_results(study):
    """Plot optimization results.

    Args:
        study: Completed Optuna study
    """
    import matplotlib.pyplot as plt

    os.makedirs("experiments/lsst/outputs", exist_ok=True)

    # Plot optimization history
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    optuna.visualization.matplotlib.plot_optimization_history(study)
    plt.title("Optimization History")

    plt.subplot(1, 3, 2)
    optuna.visualization.matplotlib.plot_parallel_coordinate(study)
    plt.title("Parallel Coordinate")

    plt.subplot(1, 3, 3)
    optuna.visualization.matplotlib.plot_param_importances(study)
    plt.title("Parameter Importances")

    plt.tight_layout()
    plt.savefig("experiments/lsst/outputs/optuna_results.png", dpi=100, bbox_inches="tight")
    plt.show()

    logger.info("Optimization plots saved to experiments/lsst/outputs/optuna_results.png")


if __name__ == "__main__":
    # Run optimization
    study = optimize_hyperparameters(n_trials=50, timeout=3600)

    # Plot results
    try:
        plot_optimization_results(study)
    except Exception as e:
        logger.warning(f"Could not plot optimization results: {e}")

    logger.info("Hyperparameter optimization completed successfully!")
    logger.info("Best results saved to experiments/lsst/outputs/best_hyperparameters.json")
