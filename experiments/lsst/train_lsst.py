#!/usr/bin/env python
"""Integrated LSST Training Script using ALP Framework

This script replicates the functionality of train_model.py but uses
the ALP framework for better modularity and extensibility.
"""

import os
import json

import matplotlib.pyplot as plt
import numpy as np
import optuna
import tensorflow as tf

from alp.data.data_reading import force_x_range_in_training
from alp.data.datasets import load_lsst_data, preprocess_lsst_data
from alp.networks.mlp import MLP
from alp.networks.uncertainty import UncertaintyQuantifier
from alp.physics.cosmo import calculate_distance_modulus_range, create_lcdm_model
from alp.utils.gpu_config import setup_tensorflow_for_training, create_safe_dataset
from alp.utils.logger_config import logger


def objective(trial, z_train, y_train, z_test, y_test):
    """Objective function for Optuna optimization.

    Parameters
    ----------
    trial : optuna.trial.Trial
        Optuna trial object
    z_train : np.ndarray
        Training redshift data
    y_train : np.ndarray
        Training target data (mu and error)
    z_test : np.ndarray
        Test redshift data
    y_test : np.ndarray
        Test target data (mu and error)

    Returns
    -------
    float
        Best validation loss achieved
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

        # Create safe datasets with parallelization disabled
        train_dataset = create_safe_dataset(z_train, y_train, batch_size=batch_size, shuffle=True)
        test_dataset = create_safe_dataset(z_test, y_test, batch_size=batch_size, shuffle=False)

        history = keras_model.fit(
            train_dataset,
            validation_data=test_dataset,
            epochs=1000,
            verbose=0,
            callbacks=[early_stopping, reduce_lr],
        )

        # Return best validation loss
        val_loss = min(history.history["val_loss"])
        logger.info(f"Trial {trial.number}: val_loss={val_loss:.4f}")

        return val_loss

    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        return float("inf")


def optimize_hyperparameters(z_train, y_train, z_test, y_test, n_trials=50, timeout=3600):
    """Run Optuna hyperparameter optimization.

    Parameters
    ----------
    z_train : np.ndarray
        Training redshift data
    y_train : np.ndarray
        Training target data (mu and error)
    z_test : np.ndarray
        Test redshift data
    y_test : np.ndarray
        Test target data (mu and error)
    n_trials : int, optional
        Number of optimization trials (default: 50)
    timeout : int, optional
        Maximum time in seconds (default: 3600)

    Returns
    -------
    dict
        Best hyperparameters from optimization
    """

    # Create study with advanced sampling and pruning
    study = optuna.create_study(
        study_name="lsst_alp_optimization",
        direction="minimize",
        sampler=optuna.samplers.NSGAIISampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    logger.info(f"Starting hyperparameter optimization with {n_trials} trials, {timeout}s timeout")

    # Create lambda function to pass data to objective
    objective_with_data = lambda trial: objective(trial, z_train, y_train, z_test, y_test)

    # Optimize with progress monitoring
    study.optimize(objective_with_data, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

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

    return trial.params


def plot_optimization_results(study):
    """Plot optimization results.

    Parameters
    ----------
    study : optuna.study.Study
        Completed Optuna study
    """

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
    plt.close()

    logger.info("Optimization plots saved to experiments/lsst/outputs/optuna_results.png")


def train_lsst_model(best_hyperparams=None):
    """Train LSST dual-output regression model using ALP framework.

    Parameters
    ----------
    best_hyperparams : dict, optional
        Best hyperparameters from optimization. If None, uses defaults.

    Returns
    -------
    tuple
        Training history, UQ results, test redshift range, and data
    """

    # Load and preprocess data
    logger.info("Loading LSST data...")
    z_data, mu_data, error_data = load_lsst_data()
    z_train, z_test, y_train, y_test, scaler = preprocess_lsst_data(z_data, mu_data, error_data)
    z_train, y_train, _ = force_x_range_in_training(
        z_train, y_train, np.concatenate([z_train, z_test]), np.concatenate([y_train, y_test])
    )

    logger.info(f"Data loaded: {len(z_train)} training samples, {len(z_test)} test samples")

    # Debug: check data ranges
    logger.info(f"Z range: {np.min(z_data):.4f} - {np.max(z_data):.4f}")
    logger.info(f"Mu range: {np.min(mu_data):.4f} - {np.max(mu_data):.4f}")
    logger.info(f"Error range: {np.min(error_data):.4f} - {np.max(error_data):.4f}")
    logger.info(f"Training mu range: {np.min(y_train[:, 0]):.4f} - {np.max(y_train[:, 0]):.4f}")

    # Use optimized hyperparameters or defaults
    if best_hyperparams is None:
        logger.info("No optimized hyperparameters provided, using defaults...")
        deep = [200, 200, 200, 200]
        dropout = 0.1
        learning_rate = 0.0001
        batch_size = 16
    else:
        logger.info("Using optimized hyperparameters from Optuna")
        deep = best_hyperparams["deep"]
        dropout = best_hyperparams["dropout"]
        learning_rate = best_hyperparams["lr"]
        batch_size = best_hyperparams["batch_size"]
        logger.info(f"  Architecture: {deep}")
        logger.info(f"  Dropout: {dropout:.4f}")
        logger.info(f"  Learning Rate: {learning_rate:.6e}")
        logger.info(f"  Batch Size: {batch_size}")

    # Create ALP model
    logger.info("Creating ALP dual-output MLP...")
    model = MLP(n_inputs=1, deep=deep, dropout=dropout, mcdropout=True, n_outputs=2)
    keras_model = model.model_tf()

    # Compile model
    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )

    # Train model
    logger.info("Training model...")
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=500, restore_best_weights=True, verbose=1
    )

    # Create safe datasets with parallelization disabled
    train_dataset = create_safe_dataset(z_train, y_train, batch_size=batch_size, shuffle=True)
    test_dataset = create_safe_dataset(z_test, y_test, batch_size=batch_size, shuffle=False)

    history = keras_model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=1000,
        verbose=2,
        callbacks=[early_stopping],
    )

    # Log training results
    final_val_loss = history.history["val_loss"][-1]
    logger.info(f"Training completed. Final validation loss: {final_val_loss:.4f}")

    # Uncertainty quantification
    logger.info("Performing MC Dropout uncertainty quantification...")
    uq = UncertaintyQuantifier(n_samples=100)

    # Generate test data
    z_min, z_max = np.min(z_data), np.max(z_data)
    z_test_range = np.linspace(z_min, z_max, 500)
    results = uq.mc_dropout_prediction(keras_model, z_test_range, scaler)

    # Save model
    os.makedirs("models", exist_ok=True)
    keras_model.save("models/alp_lsst_model.h5")
    logger.info("Model saved to models/alp_lsst_model.h5")

    return history, results, z_test_range, z_data, mu_data, error_data


def get_lcdm_predictions():
    """Get ΛCDM predictions using ALP physics module."""
    # Use default ΛCDM cosmology (H0=70, Om=0.27)
    cosmology = create_lcdm_model()

    # Calculate distance modulus over the relevant redshift range
    z_model, flcdm = calculate_distance_modulus_range(
        z_min=0.01, z_max=2.4, n_points=100, cosmology=cosmology
    )

    return z_model, flcdm


def plot_training_history(history):
    """Plot training loss history."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], "r-", label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], "g-", label="Validation Loss", linewidth=2)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title("ALP Training History", fontsize=16, fontweight="bold")

    # Save loss plot
    os.makedirs("experiments/lsst/outputs", exist_ok=True)
    plt.savefig("experiments/lsst/outputs/training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Training history plot saved to experiments/lsst/outputs/training_loss.png")


def plot_reconstruction_comparison(results, z_test_range, z_data, mu_data, error_data):
    """Plot reconstruction comparison with improved visualization.

    Only plots within the training data bounds to avoid extrapolation.
    """
    from scipy.interpolate import interp1d

    # Get training data bounds (critical to avoid extrapolation)
    z_train_min, z_train_max = np.min(z_data), np.max(z_data)
    logger.info(f"Training redshift range: [{z_train_min:.4f}, {z_train_max:.4f}]")

    # Clip z_test_range to training bounds
    valid_mask = (z_test_range >= z_train_min) & (z_test_range <= z_train_max)
    z_plot = z_test_range[valid_mask]
    results_plot_mean = results["mean"][valid_mask, 0]
    results_plot_unc = results["combined_uncertainty"][valid_mask]

    # Get ΛCDM predictions
    z_model, flcdm = get_lcdm_predictions()

    # Clip ΛCDM to training range
    lcdm_mask = (z_model >= z_train_min) & (z_model <= z_train_max)
    z_model_plot = z_model[lcdm_mask]
    flcdm_plot = flcdm[lcdm_mask]

    # Create figure with larger size
    plt.figure(figsize=(12, 8))

    # Plot observations as background scatter
    plt.errorbar(
        z_data.flatten(),
        mu_data.flatten(),
        error_data.flatten(),
        fmt="g.",
        markersize=2,
        alpha=0.4,
        capsize=2,
        capthick=1,
        ecolor="forestgreen",
        label="LSST Observations",
        zorder=1,
    )

    # Plot ΛCDM model (within training range only)
    plt.plot(z_model_plot, flcdm_plot, "b-", linewidth=3, alpha=0.8, label="ΛCDM Theory", zorder=3)

    # Plot ALP predictions with uncertainty (within training range only)
    plt.errorbar(
        z_plot,
        results_plot_mean,
        results_plot_unc,
        markersize=4,
        fmt="o",
        ecolor="red",
        capsize=3,
        capthick=2,
        elinewidth=1.5,
        alpha=0.7,
        c="darkred",
        label="ALP Reconstruction ±σ",
        zorder=4,
    )

    # Add confidence region with continuous gradient (only within training range)
    # Create smooth interpolation strictly within bounds
    z_smooth = np.linspace(z_train_min, z_train_max, 200)

    # Interpolate ALP mean and uncertainty (only within training bounds)
    interp_mean = interp1d(
        z_plot,
        results_plot_mean,
        kind="cubic",
        bounds_error=True,
        fill_value=np.nan,
    )
    interp_unc = interp1d(
        z_plot,
        results_plot_unc,
        kind="cubic",
        bounds_error=True,
        fill_value=np.nan,
    )

    mean_smooth = interp_mean(z_smooth)
    unc_smooth = interp_unc(z_smooth)

    # Plot filled confidence region (continuous color)
    plt.fill_between(
        z_smooth,
        mean_smooth - unc_smooth,
        mean_smooth + unc_smooth,
        alpha=0.2,
        color="red",
        edgecolor="none",
        label="ALP Confidence Region",
    )

    # Add smooth boundary lines for confidence region
    plt.plot(
        z_smooth,
        mean_smooth - unc_smooth,
        "r--",
        linewidth=2,
        alpha=0.6,
        label="Lower Confidence Bound",
    )
    plt.plot(
        z_smooth,
        mean_smooth + unc_smooth,
        "r--",
        linewidth=2,
        alpha=0.6,
        label="Upper Confidence Bound",
    )

    # Formatting
    plt.xlabel("Redshift z", fontsize=16, fontweight="bold")
    plt.ylabel("Distance Modulus μ(z)", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(z_train_min - 0.05, z_train_max + 0.05)

    # Dynamic ylim: extend to cover displayed data + uncertainty + margin
    y_min = np.min(results_plot_mean - results_plot_unc) - 0.5
    y_max = np.max(results_plot_mean + results_plot_unc) + 0.5
    plt.ylim(y_min, y_max)

    plt.grid(True, alpha=0.3, linestyle="--")

    # Enhanced legend
    legend = plt.legend(loc="upper left", fontsize=11, framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor("white")

    plt.title(
        "LSST Distance Modulus: Observations vs Theory vs ALP Reconstruction (Training Range Only)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Save reconstruction plot
    plt.savefig(
        "experiments/lsst/outputs/reconstruction_comparison.png",
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    logger.info(
        "Reconstruction comparison plot saved to experiments/lsst/outputs/reconstruction_comparison.png"
    )


def plot_results(history, results, z_test_range, z_data, mu_data, error_data):
    """Plot training results and predictions with separate saved plots."""

    # Create output directory
    os.makedirs("experiments/lsst/outputs", exist_ok=True)

    logger.info("Generating training history plot...")
    plot_training_history(history)

    logger.info("Generating reconstruction comparison plot...")
    plot_reconstruction_comparison(results, z_test_range, z_data, mu_data, error_data)

    # Also create a summary plot with both
    create_summary_plot(history, results, z_test_range, z_data, mu_data, error_data)

    logger.info("All plots saved to experiments/lsst/outputs/")


def create_summary_plot(history, results, z_test_range, z_data, mu_data, error_data):
    """Create a summary plot with training metrics and reconstruction.

    Only plots within the training data bounds to avoid extrapolation.
    """
    from scipy.interpolate import interp1d

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Final training loss comparison
    ax1.bar(
        ["Training", "Validation"],
        [history.history["loss"][-1], history.history["val_loss"][-1]],
        color=["red", "green"],
        alpha=0.7,
    )
    ax1.set_ylabel("Final MSE Loss", fontsize=12)
    ax1.set_title("Training Performance", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Right: Reconstruction within training range only
    z_train_min, z_train_max = np.min(z_data), np.max(z_data)

    # Clip to training range
    mask_zoom = (z_test_range >= z_train_min) & (z_test_range <= z_train_max)

    if np.any(mask_zoom):
        z_zoom = z_test_range[mask_zoom]
        results_zoom = results["mean"][mask_zoom, 0]
        unc_zoom = results["combined_uncertainty"][mask_zoom]

        # Plot zoomed region
        ax2.errorbar(
            z_zoom,
            results_zoom,
            unc_zoom,
            fmt="o",
            markersize=3,
            ecolor="red",
            capsize=2,
            elinewidth=1.5,
            alpha=0.8,
            c="darkred",
            label="ALP Reconstruction",
        )

        # Add smooth confidence region strictly within training bounds
        z_smooth = np.linspace(z_train_min, z_train_max, 100)
        interp_mean = interp1d(
            z_zoom, results_zoom, kind="cubic", bounds_error=True, fill_value=np.nan
        )
        interp_unc = interp1d(z_zoom, unc_zoom, kind="cubic", bounds_error=True, fill_value=np.nan)

        mean_smooth = interp_mean(z_smooth)
        unc_smooth = interp_unc(z_smooth)

        ax2.fill_between(
            z_smooth,
            mean_smooth - unc_smooth,
            mean_smooth + unc_smooth,
            alpha=0.3,
            color="coral",
            edgecolor="none",
        )

        # Add ΛCDM reference line only within training range
        z_model, flcdm = get_lcdm_predictions()
        z_lcdm_zoom = z_model[(z_model >= z_train_min) & (z_model <= z_train_max)]
        flcdm_zoom = flcdm[(z_model >= z_train_min) & (z_model <= z_train_max)]
        ax2.plot(z_lcdm_zoom, flcdm_zoom, "b-", linewidth=2, alpha=0.8, label="ΛCDM Theory")

    ax2.set_xlabel("Redshift z", fontsize=12)
    ax2.set_ylabel("μ(z)", fontsize=12)
    ax2.set_title(
        f"Reconstruction (z ∈ [{z_train_min:.3f}, {z_train_max:.3f}])",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig("experiments/lsst/outputs/summary_plots.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Summary plots saved to experiments/lsst/outputs/summary_plots.png")


def main():
    """Main training function."""
    logger.info("Starting LSST dual-output regression training with ALP framework")


def main():
    """Main training function."""
    logger.info("Starting LSST dual-output regression training with ALP framework")

    # Create output directories
    os.makedirs("experiments/lsst/outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Setup TensorFlow BEFORE any TensorFlow operations (critical for GPU device configuration)
    # This must be done before optimization phase to avoid "Visible devices cannot be modified" error
    logger.info("Setting up TensorFlow environment...")
    setup_tensorflow_for_training(seed=42, force_cpu=True)

    # Load data for optimization
    logger.info("Loading data for hyperparameter optimization...")
    z_data, mu_data, error_data = load_lsst_data()
    z_train, z_test, y_train, y_test, scaler = preprocess_lsst_data(z_data, mu_data, error_data)
    z_train, y_train, _ = force_x_range_in_training(
        z_train, y_train, np.concatenate([z_train, z_test]), np.concatenate([y_train, y_test])
    )

    # Run hyperparameter optimization
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Hyperparameter Optimization with Optuna NSGA-II")
    logger.info("=" * 70)
    best_hyperparams = optimize_hyperparameters(z_train, y_train, z_test, y_test, n_trials=50)
    logger.info("=" * 70 + "\n")

    # Train and evaluate with best hyperparameters
    logger.info("PHASE 2: Training with Optimized Hyperparameters")
    history, results, z_test_range, z_data, mu_data, error_data = train_lsst_model(
        best_hyperparams
    )

    # Plot results
    plot_results(history, results, z_test_range, z_data, mu_data, error_data)

    logger.info("LSST training completed successfully!")
    logger.info("Model saved: models/alp_lsst_model.h5")
    logger.info("Best hyperparameters saved: experiments/lsst/outputs/best_hyperparameters.json")
    logger.info("Results plot: experiments/lsst/outputs/lsst_alp_results.png")


if __name__ == "__main__":
    main()
