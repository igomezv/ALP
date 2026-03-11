#!/usr/bin/env python
"""Integrated LSST Training Script using ALP Framework"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from alp.data.datasets import load_lsst_data, preprocess_lsst_data
from alp.data.data_reading import force_x_range_in_training
from alp.networks.mlp import MLP
from alp.networks.uncertainty import UncertaintyQuantifier
from alp.utils.gpu_config import setup_tensorflow_for_training
from alp.utils.logger_config import logger


def plot_training_history(history, outdir):
    """Plot training loss history."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], "r-", label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], "g-", label="Validation Loss", linewidth=2)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title("ALP Training History - LSST", fontsize=16, fontweight="bold")

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "training_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Training history plot saved to experiments/lsst/outputs/training_loss.png")


def plot_reconstruction_comparison(results, z_test_range, z_data, mu_data, error_data, outdir):
    """Plot μ(z) reconstruction comparison with observations."""
    from scipy.interpolate import interp1d

    z_train_min, z_train_max = np.min(z_data), np.max(z_data)
    logger.info(f"Training redshift range: [{z_train_min:.4f}, {z_train_max:.4f}]")

    valid_mask = (z_test_range >= z_train_min) & (z_test_range <= z_train_max)
    z_plot = z_test_range[valid_mask]
    results_plot_mean = results["mean"][valid_mask, 0]
    results_plot_unc = results["combined_uncertainty"][valid_mask]

    plt.figure(figsize=(12, 8))

    plt.errorbar(
        z_data.flatten(), mu_data.flatten(), error_data.flatten(),
        fmt="g.", markersize=2, alpha=0.4, capsize=2, capthick=1, ecolor="forestgreen",
        label="LSST Observations", zorder=1,
    )

    plt.errorbar(
        z_plot, results_plot_mean, results_plot_unc,
        markersize=4, fmt="o", ecolor="red", capsize=3, capthick=2, elinewidth=1.5,
        alpha=0.7, c="darkred", label="ALP Reconstruction ±σ", zorder=4,
    )

    z_smooth = np.linspace(z_train_min, z_train_max, 200)
    interp_mean = interp1d(z_plot, results_plot_mean, kind="cubic", bounds_error=True, fill_value=np.nan)
    interp_unc = interp1d(z_plot, results_plot_unc, kind="cubic", bounds_error=True, fill_value=np.nan)

    mean_smooth = interp_mean(z_smooth)
    unc_smooth = interp_unc(z_smooth)

    plt.fill_between(z_smooth, mean_smooth - unc_smooth, mean_smooth + unc_smooth,
                     alpha=0.2, color="red", edgecolor="none", label="ALP Confidence Region")

    plt.plot(z_smooth, mean_smooth - unc_smooth, "r--", linewidth=2, alpha=0.6)
    plt.plot(z_smooth, mean_smooth + unc_smooth, "r--", linewidth=2, alpha=0.6)

    plt.xlabel("Redshift z", fontsize=16, fontweight="bold")
    plt.ylabel("μ(z) [mag]", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(z_train_min - 0.05, z_train_max + 0.05)

    y_min = np.min(results_plot_mean - results_plot_unc) - 0.5
    y_max = np.max(results_plot_mean + results_plot_unc) + 0.5
    plt.ylim(y_min, y_max)

    plt.grid(True, alpha=0.3, linestyle="--")
    legend = plt.legend(loc="upper left", fontsize=11, framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor("white")

    plt.title("μ(z): Observations vs ALP Reconstruction (Training Range Only)",
              fontsize=18, fontweight="bold", pad=20)

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "reconstruction_comparison.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("Reconstruction comparison plot saved to experiments/lsst/outputs/reconstruction_comparison.png")


def plot_summary(history, results, z_test_range, z_data, mu_data, outdir):
    """Create summary plot with training metrics and reconstruction."""
    from scipy.interpolate import interp1d

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.bar(["Training", "Validation"],
            [history.history["loss"][-1], history.history["val_loss"][-1]],
            color=["red", "green"], alpha=0.7)
    ax1.set_ylabel("Final MSE Loss", fontsize=12)
    ax1.set_title("Training Performance", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    z_train_min, z_train_max = np.min(z_data), np.max(z_data)
    mask_zoom = (z_test_range >= z_train_min) & (z_test_range <= z_train_max)

    if np.any(mask_zoom):
        z_zoom = z_test_range[mask_zoom]
        results_zoom = results["mean"][mask_zoom, 0]
        unc_zoom = results["combined_uncertainty"][mask_zoom]

        ax2.errorbar(z_zoom, results_zoom, unc_zoom, fmt="o", markersize=3,
                    ecolor="red", capsize=2, elinewidth=1.5, alpha=0.8, c="darkred",
                    label="ALP Reconstruction")

        z_smooth = np.linspace(z_train_min, z_train_max, 100)
        interp_mean = interp1d(z_zoom, results_zoom, kind="cubic", bounds_error=True, fill_value=np.nan)
        interp_unc = interp1d(z_zoom, unc_zoom, kind="cubic", bounds_error=True, fill_value=np.nan)

        mean_smooth = interp_mean(z_smooth)
        unc_smooth = interp_unc(z_smooth)

        ax2.fill_between(z_smooth, mean_smooth - unc_smooth, mean_smooth + unc_smooth,
                        alpha=0.3, color="coral", edgecolor="none")

    ax2.set_xlabel("Redshift z", fontsize=12)
    ax2.set_ylabel("μ(z) [mag]", fontsize=12)
    ax2.set_title(f"Reconstruction (z ∈ [{z_train_min:.3f}, {z_train_max:.3f}])",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "summary_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Summary plots saved to experiments/lsst/outputs/summary_plots.png")


def plot_results(history, results, z_test_range, z_data, mu_data, error_data, outdir):
    """Plot training results and predictions."""
    os.makedirs(outdir, exist_ok=True)

    logger.info("Generating training history plot...")
    plot_training_history(history, outdir)

    logger.info("Generating reconstruction comparison plot...")
    plot_reconstruction_comparison(results, z_test_range, z_data, mu_data, error_data, outdir)

    plot_summary(history, results, z_test_range, z_data, mu_data, outdir)

    logger.info("All plots saved to experiments/lsst/outputs/")


def main():
    """Train LSST dual-output regression model using ALP framework."""
    
    # Setup directories
    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Setup TensorFlow
    setup_tensorflow_for_training(seed=42, force_cpu=True)
    
    # Load and preprocess data
    logger.info("Loading LSST data...")
    z_data, mu_data, error_data = load_lsst_data()
    z_train, z_test, y_train, y_test, scaler = preprocess_lsst_data(z_data, mu_data, error_data)
    z_train, y_train, _ = force_x_range_in_training(
        z_train, y_train, np.concatenate([z_train, z_test]), np.concatenate([y_train, y_test])
    )
    
    logger.info(f"Data loaded: {len(z_train)} training samples, {len(z_test)} test samples")
    logger.info(f"Z range: {np.min(z_data):.4f} - {np.max(z_data):.4f}")
    logger.info(f"μ(z) range: {np.min(mu_data):.4f} - {np.max(mu_data):.4f}")
    logger.info(f"Error range: {np.min(error_data):.4f} - {np.max(error_data):.4f}")
    logger.info(f"Training μ(z) range: {np.min(y_train[:, 0]):.4f} - {np.max(y_train[:, 0]):.4f}")
    
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
    logger.info("Training model with optimized hyperparameters...")
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
    
    final_val_loss = history.history["val_loss"][-1]
    logger.info(f"Training completed. Final validation loss: {final_val_loss:.4f}")
    
    # Uncertainty quantification
    logger.info("Performing MC Dropout uncertainty quantification...")
    uq = UncertaintyQuantifier(n_samples=MC_DROPOUT_SAMPLES)
    z_test_range = np.linspace(np.min(z_data), np.max(z_data), 500)
    results = uq.mc_dropout_prediction(keras_model, z_test_range, scaler)
    
    # Save model
    os.makedirs("models", exist_ok=True)
    keras_model.save("models/alp_lsst_model.h5")
    logger.info("Model saved to models/alp_lsst_model.h5")
    
    # Generate plots
    plot_results(history, results, z_test_range, z_data, mu_data, error_data, outdir)
    
    logger.info("LSST training completed successfully!")
    logger.info("Model saved: models/alp_lsst_model.h5")
    logger.info("Results saved to experiments/lsst/outputs/")


if __name__ == "__main__":
    main()
