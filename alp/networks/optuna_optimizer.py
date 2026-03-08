#!/usr/bin/env python

"""
Astro Layer Perceptron Networks Module - Hyperparameter Optimization
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
Easy-to-use Optuna wrapper for hyperparameter optimization
of ALP neural networks with minimal code required.
"""

import os
import json
import optuna
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Callable, Optional, List, Union

from .mlp import MLP
from ..utils.logger_config import logger


class OptunaOptimizer:
    """
    Simplified Optuna wrapper for ALP hyperparameter optimization.

    Provides an easy interface to optimize neural network hyperparameters
    with minimal setup required. Works with any ALP network architecture.

    Parameters
    ----------
    model_class : class
        ALP model class to optimize (e.g., MLP)
    model_kwargs : dict
        Fixed parameters for the model (e.g., n_inputs, n_outputs)
    data_loader : callable
        Function that returns (X_train, X_val, y_train, y_val)
    directions : str, optional
        Optimization direction: 'minimize' or 'maximize' (default 'minimize')
    """

    def __init__(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        data_loader: Callable,
        direction: str = "minimize",
    ):
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.data_loader = data_loader
        self.direction = "minimize" if direction == "minimize" else "maximize"

        # Load data once to validate
        try:
            self.X_train, self.X_val, self.y_train, self.y_val = data_loader()
            logger.info(f"Data loaded: {len(self.X_train)} train, {len(self.X_val)} val samples")
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise

    def suggest_hyperparameters(self, trial: optuna.Trial) -> Dict[str, Any]:
        """
        Suggest hyperparameters for the trial.

        Override this method to customize hyperparameter search space.

        Parameters
        ----------
        trial : optuna.Trial
            Current optimization trial

        Returns
        -------
        dict
            Dictionary of suggested hyperparameters
        """
        return {
            "deep": trial.suggest_categorical(
                "deep",
                [
                    [100, 100, 100],
                    [200, 200, 200, 200],
                    [300, 300, 300],
                    [128, 256, 128, 64],
                    [64, 128, 256, 128, 64],
                ],
            ),
            "dropout": trial.suggest_float("dropout", 0.05, 0.3),
            "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        }

    def create_model(self, hyperparams: Dict[str, Any]) -> tf.keras.Model:
        """
        Create model with given hyperparameters.

        Override this method to customize model creation.

        Parameters
        ----------
        hyperparams : dict
            Hyperparameters for the model

        Returns
        -------
        tf.keras.Model
            Compiled Keras model
        """
        # Create model
        model = self.model_class(
            **self.model_kwargs,
            **{k: v for k, v in hyperparams.items() if k in ["deep", "dropout"]},
        )
        keras_model = model.model_tf()

        # Compile model
        keras_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparams["lr"]), loss="mse"
        )

        return keras_model

    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.

        Parameters
        ----------
        trial : optuna.Trial
            Current optimization trial

        Returns
        -------
        float
            Objective value (validation loss by default)
        """
        try:
            # Suggest hyperparameters
            hyperparams = self.suggest_hyperparameters(trial)

            logger.info(f"Trial {trial.number}: {hyperparams}")

            # Create model
            model = self.create_model(hyperparams)

            # Setup callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=100, restore_best_weights=True, verbose=0
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss", factor=0.5, patience=50, min_lr=1e-7, verbose=0
                ),
            ]

            # Train model
            history = model.fit(
                self.X_train,
                self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=1000,  # Reduced for optimization speed
                batch_size=hyperparams["batch_size"],
                verbose=0,
                callbacks=callbacks,
            )

            # Get best validation loss
            val_loss = min(history.history["val_loss"])
            logger.info(f"Trial {trial.number}: val_loss={val_loss:.4f}")

            return val_loss

        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            return float("inf") if self.direction == "minimize" else float("-inf")

    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: str = "alp_optimization",
        sampler: Optional[optuna.samplers.BaseSampler] = None,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        output_dir: str = "outputs",
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Parameters
        ----------
        n_trials : int, optional
            Number of optimization trials (default 50)
        timeout : int, optional
            Maximum time in seconds (default None)
        study_name : str, optional
            Name of the study (default 'alp_optimization')
        sampler : optuna.samplers.BaseSampler, optional
            Custom sampler (default NSGAIISampler)
        pruner : optuna.pruners.BasePruner, optional
            Custom pruner (default MedianPruner)
        output_dir : str, optional
            Directory to save results (default 'outputs')

        Returns
        -------
        optuna.Study
            Completed study object
        """
        # Create study
        if sampler is None:
            sampler = optuna.samplers.NSGAIISampler(seed=42)
        if pruner is None:
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        study = optuna.create_study(
            study_name=study_name, direction=self.direction, sampler=sampler, pruner=pruner
        )

        logger.info(f"Starting optimization: {n_trials} trials, {timeout}s timeout")

        # Optimize
        study.optimize(self.objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        # Log results
        logger.info("Optimization completed!")
        trial = study.best_trial
        logger.info(f"Best value: {trial.value:.6f}")
        logger.info("Best hyperparameters:")
        for key, value in trial.params.items():
            logger.info(f"  {key}: {value}")

        # Save results
        self._save_results(study, trial, output_dir)

        return study

    def _save_results(self, study: optuna.Study, best_trial: optuna.Trial, output_dir: str):
        """Save optimization results to files."""
        os.makedirs(output_dir, exist_ok=True)

        # Save best hyperparameters
        best_params = {
            "best_value": float(best_trial.value),
            "best_hyperparameters": best_trial.params,
            "trial_number": best_trial.number,
            "direction": self.direction,
        }

        with open(os.path.join(output_dir, "best_hyperparameters.json"), "w") as f:
            json.dump(best_params, f, indent=2)

        # Save study summary
        study_data = {
            "best_trial": {
                "number": best_trial.number,
                "value": float(best_trial.value),
                "params": best_trial.params,
            },
            "n_trials": len(study.trials),
            "study_name": study.study_name,
            "direction": self.direction,
        }

        with open(os.path.join(output_dir, "study_summary.json"), "w") as f:
            json.dump(study_data, f, indent=2)

        logger.info(f"Results saved to {output_dir}/")

    def plot_results(self, study: optuna.Study, output_dir: str = "outputs"):
        """
        Plot optimization results.

        Parameters
        ----------
        study : optuna.Study
            Completed study object
        output_dir : str, optional
            Directory to save plots (default 'outputs')
        """
        try:
            import matplotlib.pyplot as plt

            os.makedirs(output_dir, exist_ok=True)

            # Create plots
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
            plt.savefig(
                os.path.join(output_dir, "optuna_results.png"), dpi=100, bbox_inches="tight"
            )
            plt.close()

            logger.info(f"Optimization plots saved to {output_dir}/optuna_results.png")

        except Exception as e:
            logger.warning(f"Could not plot optimization results: {e}")


def quick_optimize(
    model_class,
    model_kwargs: Dict[str, Any],
    data_loader: Callable,
    n_trials: int = 50,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """
    Quick optimization function with minimal setup.

    Parameters
    ----------
    model_class : class
        ALP model class to optimize (e.g., MLP)
    model_kwargs : dict
        Fixed parameters for the model (e.g., n_inputs, n_outputs)
    data_loader : callable
        Function that returns (X_train, X_val, y_train, y_val)
    n_trials : int, optional
        Number of optimization trials (default 50)
    output_dir : str, optional
        Directory to save results (default 'outputs')

    Returns
    -------
    dict
        Best hyperparameters and value
    """
    optimizer = OptunaOptimizer(model_class, model_kwargs, data_loader)
    study = optimizer.optimize(n_trials=n_trials, output_dir=output_dir)
    optimizer.plot_results(study, output_dir)

    return {
        "best_params": study.best_trial.params,
        "best_value": study.best_trial.value,
        "study": study,
    }
