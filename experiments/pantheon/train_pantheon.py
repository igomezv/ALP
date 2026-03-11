#!/usr/bin/env python
"""Training Script for Pantheon SNIa using ALP Framework"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from alp.data.datasets import load_pantheon_data, preprocess_pantheon_data
from alp.networks.mlp import MLP
from alp.networks.uncertainty import UncertaintyQuantifier
from alp.physics.cosmo import calculate_distance_modulus_range, create_lcdm_model
from alp.utils.gpu_config import setup_tensorflow_for_training
from alp.utils.logger_config import logger


def train_pantheon_model(z_train, y_train, z_test, y_test, learning_rate=0.0001,
                         batch_size=16, dropout=0.1, architecture=None, epochs=1000):
    """Train Pantheon dual-output regression model using ALP framework."""
    if architecture is None:
        architecture = [200, 200, 200, 200]
    
    logger.info("Creating ALP dual-output MLP for Pantheon data...")
    model = MLP(
        n_inputs=1,
        deep=architecture,
        dropout=dropout,
        mcdropout=True,
        n_outputs=2,
    )
    keras_model = model.model_tf()

    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse"
    )

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

    final_val_loss = history.history["val_loss"][-1]
    logger.info(f"Training completed. Final validation loss: {final_val_loss:.4f}")

    return keras_model, history


def plot_training_history(history, outdir):
    """Plot training loss history."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history["loss"], "r-", label="Training Loss", linewidth=2)
    plt.plot(history.history["val_loss"], "g-", label="Validation Loss", linewidth=2)
    plt.ylabel("MSE Loss", fontsize=14)
    plt.xlabel("Epoch", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.title("ALP Training History - Pantheon SNIa", fontsize=16, fontweight="bold")

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "training_loss.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Training history plot saved to experiments/pantheon/outputs/training_loss.png")


def plot_reconstruction_comparison(results, z_test_range, z_data, mb_data, dmb_data, outdir):
    """Plot Hubble diagram reconstruction comparison with ΛCDM."""
    z_train_min, z_train_max = np.min(z_data), np.max(z_data)
    logger.info(f"Training redshift range: [{z_train_min:.4f}, {z_train_max:.4f}]")

    valid_mask = (z_test_range >= z_train_min) & (z_test_range <= z_train_max)
    z_plot = z_test_range[valid_mask]
    results_plot_mean = results["mean"][valid_mask, 0]
    results_plot_unc = results["combined_uncertainty"][valid_mask]

    # Get ΛCDM predictions
    cosmology = create_lcdm_model()
    z_lcdm, mu_lcdm = calculate_distance_modulus_range(
        z_min=z_train_min, z_max=z_train_max, n_points=200, cosmology=cosmology
    )

    plt.figure(figsize=(12, 8))

    plt.errorbar(
        z_data.flatten(), mb_data.flatten(), dmb_data.flatten(),
        fmt="g.", markersize=2, alpha=0.4, capsize=2, capthick=1, ecolor="forestgreen",
        label="Pantheon Observations", zorder=1,
    )

    plt.plot(z_lcdm, mu_lcdm, "b-", linewidth=3, alpha=0.8, label="ΛCDM Theory", zorder=3)

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
    plt.ylabel("Distance Modulus μ", fontsize=16, fontweight="bold")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(z_train_min - 0.05, z_train_max + 0.05)

    y_min = np.min(results_plot_mean - results_plot_unc) - 1
    y_max = np.max(results_plot_mean + results_plot_unc) + 1
    plt.ylim(y_min, y_max)

    plt.grid(True, alpha=0.3, linestyle="--")
    legend = plt.legend(loc="upper right", fontsize=11, framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor("white")

    plt.title("Pantheon: Observations vs Theory vs ALP Reconstruction (Training Range Only)",
              fontsize=18, fontweight="bold", pad=20)

    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "reconstruction_comparison.png"), dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info("Reconstruction comparison plot saved to experiments/pantheon/outputs/reconstruction_comparison.png")


def plot_summary(history, results, z_test_range, z_data, mb_data, outdir):
    """Create summary plot with training metrics and reconstruction."""
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

        cosmology = create_lcdm_model()
        z_lcdm, mu_lcdm = calculate_distance_modulus_range(
            z_min=z_train_min, z_max=z_train_max, n_points=100, cosmology=cosmology
        )
        ax2.plot(z_lcdm, mu_lcdm, "b-", linewidth=2, alpha=0.8, label="ΛCDM Theory")

    ax2.set_xlabel("Redshift z", fontsize=12)
    ax2.set_ylabel("Distance Modulus μ", fontsize=12)
    ax2.set_title(f"Reconstruction (z ∈ [{z_train_min:.3f}, {z_train_max:.3f}])",
                  fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    plt.tight_layout()
    os.makedirs(outdir, exist_ok=True)
    plt.savefig(os.path.join(outdir, "summary_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Summary plots saved to experiments/pantheon/outputs/summary_plots.png")


def main():
    """Main training function for Pantheon data."""
    logger.info("Starting Pantheon SNIa training with ALP framework")

    here = os.path.dirname(__file__)
    outdir = os.path.join(here, "outputs")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    setup_tensorflow_for_training(seed=42, force_cpu=True)

    logger.info("Loading Pantheon data...")
    z_data, mb_data, dmb_data = load_pantheon_data("data/pantheon_lcparam_full_long_zhel.txt")
    z_train, z_test, y_train, y_test, scaler = preprocess_pantheon_data(z_data, mb_data, dmb_data)

    logger.info(f"Data loaded: {len(z_train)} training samples, {len(z_test)} test samples")
    logger.info(f"Z range: {np.min(z_data):.4f} - {np.max(z_data):.4f}")
    logger.info(f"mb range: {np.min(mb_data):.4f} - {np.max(mb_data):.4f}")
    logger.info(f"dmb range: {np.min(dmb_data):.4f} - {np.max(dmb_data):.4f}")
    logger.info(f"Training mb range: {np.min(y_train[:, 0]):.4f} - {np.max(y_train[:, 0]):.4f}")

    logger.info("Training model with default hyperparameters...")
    model, history = train_pantheon_model(
        z_train, y_train, z_test, y_test,
        learning_rate=0.0001,
        batch_size=16,
        dropout=0.1,
        architecture=[200, 200, 200, 200],
        epochs=1000,
    )

    logger.info("Performing MC Dropout uncertainty quantification...")
    uq = UncertaintyQuantifier(n_samples=100)

    z_min, z_max = np.min(z_data), np.max(z_data)
    z_test_range = np.linspace(z_min, z_max, 500)
    results = uq.mc_dropout_prediction(model, z_test_range, scaler)

    os.makedirs("models", exist_ok=True)
    model.save("models/alp_pantheon_model.h5")
    logger.info("Model saved to models/alp_pantheon_model.h5")

    plot_training_history(history, outdir)
    plot_reconstruction_comparison(results, z_test_range, z_data, mb_data, dmb_data, outdir)
    plot_summary(history, results, z_test_range, z_data, mb_data, outdir)

    logger.info("Pantheon training completed successfully!")
    logger.info("Model saved: models/alp_pantheon_model.h5")
    logger.info("Results saved to experiments/pantheon/outputs/")


if __name__ == "__main__":
    main()
