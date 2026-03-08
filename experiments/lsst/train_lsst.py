#!/usr/bin/env python
"""Integrated LSST Training Script using ALP Framework

This script replicates the functionality of train_model.py but uses
the ALP framework for better modularity and extensibility.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from alp.data.data_reading import force_x_range_in_training
from alp.data.datasets import load_lsst_data, preprocess_lsst_data
from alp.networks.mlp import MLP
from alp.networks.uncertainty import UncertaintyQuantifier
from alp.physics.cosmo import calculate_distance_modulus_range, create_lcdm_model
from alp.utils.gpu_config import setup_tensorflow_for_training
from alp.utils.logger_config import logger


def train_lsst_model():
    """Train LSST dual-output regression model using ALP framework."""

    # Setup TensorFlow with GPU-safe configuration (use CPU to avoid CUDA PTX issues)
    setup_tensorflow_for_training(seed=42, force_cpu=True)

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

    # Create ALP model
    logger.info("Creating ALP dual-output MLP...")
    model = MLP(n_inputs=1, deep=[200, 200, 200, 200], dropout=0.1, mcdropout=True, n_outputs=2)
    keras_model = model.model_tf()

    # Compile model
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
    """Plot reconstruction comparison with improved visualization."""

    # Get ΛCDM predictions
    z_model, flcdm = get_lcdm_predictions()

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

    # Plot ΛCDM model
    plt.plot(z_model, flcdm, "b-", linewidth=3, alpha=0.8, label="ΛCDM Theory", zorder=3)

    # Plot ALP predictions with uncertainty
    plt.errorbar(
        z_test_range,
        results["mean"][:, 0],
        results["combined_uncertainty"],
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

    # Add confidence region with continuous gradient
    from scipy.interpolate import interp1d

    # Create smooth interpolation for better visualization
    # Use slightly narrower range to avoid interpolation bounds errors
    z_smooth = np.linspace(z_test_range.min() * 1.001, z_test_range.max() * 0.999, 200)

    # Interpolate ALP mean and uncertainty with bounds_error=False for safety
    interp_mean = interp1d(
        z_test_range,
        results["mean"][:, 0],
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interp_unc = interp1d(
        z_test_range,
        results["combined_uncertainty"],
        kind="cubic",
        bounds_error=False,
        fill_value="extrapolate",
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
    plt.xlim(0, 1.3)
    plt.ylim(33, 46)
    plt.grid(True, alpha=0.3, linestyle="--")

    # Enhanced legend
    legend = plt.legend(loc="upper left", fontsize=11, framealpha=0.9, fancybox=True, shadow=True)
    legend.get_frame().set_facecolor("white")

    plt.title(
        "LSST Distance Modulus: Observations vs Theory vs ALP Reconstruction",
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
    """Create a summary plot with training metrics and reconstruction."""

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

    # Right: Reconstruction zoom-in (limited region)
    z_model, flcdm = get_lcdm_predictions()

    # Zoom to region with good data coverage
    z_min_zoom, z_max_zoom = 0.1, 1.0
    mask_zoom = (z_test_range >= z_min_zoom) & (z_test_range <= z_max_zoom)

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

        # Add smooth confidence region for zoomed area
        from scipy.interpolate import interp1d

        z_smooth = np.linspace(z_min_zoom, z_max_zoom, 100)
        interp_mean = interp1d(
            z_zoom, results_zoom, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )
        interp_unc = interp1d(
            z_zoom, unc_zoom, kind="cubic", bounds_error=False, fill_value="extrapolate"
        )

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

        # Add ΛCDM reference line
        z_lcdm_zoom = z_model[(z_model >= z_min_zoom) & (z_model <= z_max_zoom)]
        flcdm_zoom = flcdm[(z_model >= z_min_zoom) & (z_model <= z_max_zoom)]
        ax2.plot(z_lcdm_zoom, flcdm_zoom, "b-", linewidth=2, alpha=0.8, label="ΛCDM Theory")

    ax2.set_xlabel("Redshift z", fontsize=12)
    ax2.set_ylabel("μ(z)", fontsize=12)
    ax2.set_title(
        f"Reconstruction Detail (z ∈ [{z_min_zoom}, {z_max_zoom}])", fontsize=14, fontweight="bold"
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

    # Create output directories
    os.makedirs("experiments/lsst/outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Train and evaluate
    history, results, z_test_range, z_data, mu_data, error_data = train_lsst_model()

    # Plot results
    plot_results(history, results, z_test_range, z_data, mu_data, error_data)

    logger.info("LSST training completed successfully!")
    logger.info("Model saved: models/alp_lsst_model.h5")
    logger.info("Results plot: experiments/lsst/outputs/lsst_alp_results.png")


if __name__ == "__main__":
    main()
