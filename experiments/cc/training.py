#!/usr/bin/env python
"""Integrated CC (Cosmic Chronometers) Training Script using ALP Framework

This script replicates the functionality of the LSST training but adapted
for cosmic chronometers H(z) data analysis.

FEATURES:
- Hyperparameter Optimization: Uses Optuna with NSGA-II genetic algorithm
  to find optimal learning_rate, batch_size, dropout_rate, and layer_width.
  The optimization phase minimizes both validation loss and model complexity.

- Multi-Objective Optimization: Balances:
  1. Validation loss minimization (primary objective)
  2. Model parameter reduction (secondary objective)

- Two-Phase Training:
  PHASE 1: Hyperparameter optimization (configurable number of trials)
  PHASE 2: Final training with best hyperparameters

OPTIMIZATION PARAMETERS:
  - learning_rate: [1e-5, 1e-3] (log scale)
  - batch_size: [8, 16, 24, 32]
  - dropout_rate: [0.05, 0.10, 0.15, ..., 0.30]
  - layer_width: [64, 96, 128, ..., 256]

TUNING THE OPTIMIZATION:
  To adjust the number of trials, edit the n_trials parameter in main():
  - n_trials=2:  ~2-3 minutes (quick test)
  - n_trials=5:  ~5-7 minutes (fast)
  - n_trials=10: ~15-20 minutes (balanced)
  - n_trials=50: ~2+ hours (comprehensive, recommended for production)
"""

import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import NSGAIISampler

from alp.data.data_reading import force_grid_endpoints, force_x_range_in_training
from alp.data.datasets import load_hz31_data, preprocess_cc_data
from alp.networks.mlp import MLP
from alp.networks.uncertainty import UncertaintyQuantifier
from alp.physics.cosmo import get_lcdm_hubble
from alp.utils.gpu_config import setup_tensorflow_for_training
from alp.utils.logger_config import logger


# ============================================================================
# HYPERPARAMETER OPTIMIZATION WITH OPTUNA + NSGA-II
# ============================================================================
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
                epochs=200,  # Reduced for faster optimization trials
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
    best_trial = study.best_trials[0]  # First trial is best by first objective

    logger.info(f"\nOptimization completed!")
    logger.info(f"Best trial number: {best_trial.number}")
    logger.info(f"Best validation loss: {best_trial.values[0]:.4f}")
    logger.info(f"Best model complexity: {best_trial.values[1]:.2f}k parameters")
    logger.info(f"Best hyperparameters:")
    for key, value in best_trial.params.items():
        logger.info(f"  {key}: {value}")

    return best_trial.params


def train_cc_model(
    z_train,
    y_train,
    z_test,
    y_test,
    learning_rate=0.0001,
    batch_size=16,
    dropout_rate=0.1,
    layer_width=200,
    epochs=1000,
):
    """Train CC dual-output regression model using ALP framework.

    Parameters
    ----------
    z_train : np.ndarray
        Training redshift values (N, 1)
    y_train : np.ndarray
        Training H(z) values (N, 2) with [H(z), error]
    z_test : np.ndarray
        Test redshift values (M, 1)
    y_test : np.ndarray
        Test H(z) values (M, 2) with [H(z), error]
    learning_rate : float
        Adam optimizer learning rate (default: 0.0001)
    batch_size : int
        Training batch size (default: 16)
    dropout_rate : float
        MC Dropout rate (default: 0.1)
    layer_width : int
        Width of hidden layers (default: 200)
    epochs : int
        Maximum number of training epochs (default: 1000)

    Returns
    -------
    tuple
        (model, history) - Trained model and training history
    """
    # Create ALP model with provided hyperparameters
    logger.info("Creating ALP dual-output MLP for CC data...")
    model = MLP(
        n_inputs=1,
        deep=[layer_width, layer_width, layer_width, layer_width],
        dropout=dropout_rate,
        mcdropout=True,
        n_outputs=2,
    )
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

    history = keras_model.fit(
        z_train,
        y_train,
        validation_data=(z_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
        callbacks=[early_stopping],
    )

    # Log training results
    final_val_loss = history.history["val_loss"][-1]
    logger.info(f"Training completed. Final validation loss: {final_val_loss:.4f}")

    return keras_model, history


def plot_training_history(history, outdir):
    """Plot training loss history.

    Parameters
    ----------
    history : tf.keras.History
        Training history object from model.fit()
    outdir : str
        Output directory for saving plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], "r-", label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], "g-", label="Validation Loss", linewidth=2)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title("ALP Training History - Cosmic Chronometers", fontsize=16, fontweight="bold")

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "training_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Training history plot saved to experiments/cc/outputs/training_loss.png")


def plot_reconstruction_comparison(results, z_test_range, z_data, hz_data, error_data, outdir):
    """Plot H(z) reconstruction comparison with ΛCDM.

    Parameters
    ----------
    results : dict
        Dictionary with 'mean' and 'combined_uncertainty' keys from UncertaintyQuantifier
    z_test_range : np.ndarray
        Redshift range for predictions (within training bounds)
    z_data : np.ndarray
        Original redshift data
    hz_data : np.ndarray
        Original H(z) data
    error_data : np.ndarray
        Original error data
    outdir : str
        Output directory for saving plot
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

    # Get ΛCDM predictions only within training range
    hz_lcdm = get_lcdm_hubble(z_plot)

    plt.figure(figsize=(12, 8))

    # Plot observations as background scatter
    plt.errorbar(
        z_data.flatten(),
        hz_data.flatten(),
        error_data.flatten(),
        fmt="g.",
        markersize=2,
        alpha=0.4,
        capsize=2,
        capthick=1,
        ecolor="forestgreen",
        label="CC Observations",
        zorder=1,
    )

    # Plot ΛCDM model within training range only
    plt.plot(z_plot, hz_lcdm, "b-", linewidth=3, alpha=0.8, label="ΛCDM Theory", zorder=3)

    # Plot ALP predictions with uncertainty (only within training range)
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

    # Plot filled confidence region
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
    plt.ylabel("H(z) [km/s/Mpc]", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(z_train_min - 0.05, z_train_max + 0.05)

    # Dynamic ylim: extend to cover displayed data + uncertainty + margin
    y_min = np.min(results_plot_mean - results_plot_unc) - 10
    y_max = np.max(results_plot_mean + results_plot_unc) + 10
    plt.ylim(y_min, y_max)

    plt.grid(True, alpha=0.3, linestyle="--")

    # Enhanced legend
    legend = plt.legend(loc="upper right", fontsize=11, framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor("white")

    plt.title(
        "H(z): Observations vs Theory vs ALP Reconstruction (Training Range Only)",
        fontsize=18,
        fontweight="bold",
        pad=20,
    )

    # Save reconstruction plot
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(
        os.path.join(outdir, "reconstruction_comparison.png"),
        dpi=150,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close()

    logger.info(
        "Reconstruction comparison plot saved to experiments/cc/outputs/reconstruction_comparison.png"
    )


def plot_summary(history, results, z_test_range, z_data, hz_data, outdir):
    """Create summary plot with training metrics and reconstruction.

    Parameters
    ----------
    history : tf.keras.History
        Training history object
    results : dict
        Dictionary with 'mean' and 'combined_uncertainty' keys
    z_test_range : np.ndarray
        Redshift range for predictions
    z_data : np.ndarray
        Original redshift data
    hz_data : np.ndarray
        Original H(z) data
    outdir : str
        Output directory for saving plot
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
        hz_lcdm_zoom = get_lcdm_hubble(z_smooth)
        ax2.plot(z_smooth, hz_lcdm_zoom, "b-", linewidth=2, alpha=0.8, label="ΛCDM Theory")

    ax2.set_xlabel("Redshift z", fontsize=12)
    ax2.set_ylabel("H(z) [km/s/Mpc]", fontsize=12)
    ax2.set_title(
        f"Reconstruction (z ∈ [{z_train_min:.3f}, {z_train_max:.3f}])",
        fontsize=14,
        fontweight="bold",
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "summary_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Summary plots saved to experiments/cc/outputs/summary_plots.png")


def plot_results(history, results, z_test_range, z_data, hz_data, error_data, outdir):
    """Plot training results and predictions with separate saved plots.

    Parameters
    ----------
    history : tf.keras.History
        Training history
    results : dict
        Uncertainty quantification results
    z_test_range : np.ndarray
        Redshift range for predictions
    z_data : np.ndarray
        Original redshift data
    hz_data : np.ndarray
        Original H(z) data
    error_data : np.ndarray
        Original error data
    outdir : str
        Output directory
    """
    # Create output directory
    os.makedirs(outdir, exist_ok=True)

    logger.info("Generating training history plot...")
    plot_training_history(history, outdir)

    logger.info("Generating reconstruction comparison plot...")
    plot_reconstruction_comparison(results, z_test_range, z_data, hz_data, error_data, outdir)

    # Also create a summary plot with both
    plot_summary(history, results, z_test_range, z_data, hz_data, outdir)

    logger.info("All plots saved to experiments/cc/outputs/")


def main():
    """Main training function for CC data."""
    logger.info("Starting CC (Cosmic Chronometers) training with ALP framework")

    # Create output directories
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Setup TensorFlow with GPU-safe configuration
    setup_tensorflow_for_training(seed=42, force_cpu=True)

    # Load and preprocess data
    logger.info("Loading CC H(z) data...")
    data_file = os.path.join(here, "..", "..", "data", "Hz31.txt")
    z_data, hz_data, error_data = load_hz31_data(data_file)

    z_train, z_test, y_train, y_test, scaler = preprocess_cc_data(z_data, hz_data, error_data)

    logger.info(f"Data loaded: {len(z_train)} training samples, {len(z_test)} test samples")

    # Debug: check data ranges
    logger.info(f"Z range: {np.min(z_data):.4f} - {np.max(z_data):.4f}")
    logger.info(f"H(z) range: {np.min(hz_data):.4f} - {np.max(hz_data):.4f}")
    logger.info(f"Error range: {np.min(error_data):.4f} - {np.max(error_data):.4f}")
    logger.info(f"Training H(z) range: {np.min(y_train[:, 0]):.4f} - {np.max(y_train[:, 0]):.4f}")

    # ========================================================================
    # HYPERPARAMETER OPTIMIZATION
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: Hyperparameter Optimization with Optuna NSGA-II")
    logger.info("=" * 70)
    best_hyperparams = optimize_hyperparameters(z_train, y_train, z_test, y_test, n_trials=2)
    logger.info("=" * 70 + "\n")

    # ========================================================================
    # TRAINING WITH OPTIMIZED HYPERPARAMETERS
    # ========================================================================
    logger.info("PHASE 2: Training with Optimized Hyperparameters")
    logger.info(f"Learning rate: {best_hyperparams['learning_rate']:.2e}")
    logger.info(f"Batch size: {best_hyperparams['batch_size']}")
    logger.info(f"Dropout rate: {best_hyperparams['dropout_rate']:.3f}")
    logger.info(f"Layer width: {best_hyperparams['layer_width']}")
    logger.info("=" * 70)

    # Train model with optimized hyperparameters
    model, history = train_cc_model(
        z_train,
        y_train,
        z_test,
        y_test,
        learning_rate=best_hyperparams["learning_rate"],
        batch_size=best_hyperparams["batch_size"],
        dropout_rate=best_hyperparams["dropout_rate"],
        layer_width=best_hyperparams["layer_width"],
        epochs=1000,
    )

    # Uncertainty quantification
    logger.info("Performing MC Dropout uncertainty quantification...")
    uq = UncertaintyQuantifier(n_samples=100)

    # Generate test data
    z_min, z_max = np.min(z_data), np.max(z_data)
    z_test_range = np.linspace(z_min, z_max, 500)
    results = uq.mc_dropout_prediction(model, z_test_range, scaler)

    # Save model
    os.makedirs("models", exist_ok=True)
    model.save("models/alp_cc_model.h5")
    logger.info("Model saved to models/alp_cc_model.h5")

    # Plot results
    plot_results(history, results, z_test_range, z_data, hz_data, error_data, outdir)

    logger.info("CC training completed successfully!")
    logger.info("Model saved: models/alp_cc_model.h5")
    logger.info("Results saved to experiments/cc/outputs/")


if __name__ == "__main__":
    main()
