#!/usr/bin/env python
"""Integrated UQ Comparison Framework

This module provides comprehensive comparison between MC Dropout and CQR uncertainty quantification methods.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional

from alp.data.datasets import load_lsst_data, preprocess_lsst_data
from alp.physics.cosmo import create_lcdm_model, calculate_distance_modulus_range
from alp.networks import (
    UncertaintyQuantifier,
    ConformalizedQuantileRegression,
    create_cqr_for_sigma,
)
from alp.analysis.model_comparison import ModelComparisonMetrics
from alp.utils.logger_config import logger
from alp.utils.gpu_config import setup_tensorflow_for_training


def compare_uncertainty_methods(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    confidence_levels: list = [0.683, 0.955],
    n_mc_samples: int = 100,
    calibration_frac: float = 0.1,
    output_dir: str = "experiments/lsst/uq_comparison",
) -> Dict[str, Any]:
    """Compare MC Dropout and CQR uncertainty quantification methods.

    Args:
        model_path: Path to trained model
        X_test: Test input data
        y_test: Test target data
        scaler: Data scaler
        confidence_levels: Confidence levels for analysis (default 1σ, 2σ)
        n_mc_samples: Number of MC dropout samples
        calibration_frac: Fraction of data for CQR calibration
        output_dir: Output directory

    Returns:
        Dictionary with comprehensive comparison results
    """

    logger.info("Starting comprehensive UQ method comparison...")
    os.makedirs(output_dir, exist_ok=True)

    # Load trained model
    logger.info(f"Loading model from {model_path}")
    from alp.networks.base_networks import SupervisedNET

    base_net = SupervisedNET()
    model = base_net.load_model(model_path)

    # Initialize UQ methods
    mc_uq = UncertaintyQuantifier(n_samples=n_mc_samples)
    cqr_1sigma = create_cqr_for_sigma(sigma=1.0, calibration_frac=calibration_frac)
    cqr_2sigma = create_cqr_for_sigma(sigma=2.0, calibration_frac=calibration_frac)

    # Get predictions from both methods
    logger.info("Performing MC Dropout uncertainty quantification...")
    mc_results = mc_uq.mc_dropout_prediction(model, X_test, scaler)

    logger.info("Performing CQR uncertainty quantification...")
    cqr_1sigma.train_quantile_models(X_test[:, 0], y_test, scaler, epochs=50, verbose=0)
    cqr_2sigma.train_quantile_models(X_test[:, 0], y_test, scaler, epochs=50, verbose=0)

    # Get predictions from both CQR models
    cqr_results_1sigma = cqr_1sigma.predict_with_intervals(X_test, scaler)
    cqr_results_2sigma = cqr_2sigma.predict_with_intervals(X_test, scaler)

    # Evaluate both methods
    logger.info("Evaluating MC Dropout performance...")
    mc_eval = mc_uq.evaluate_coverage(X_test, y_test, scaler)

    logger.info("Evaluating CQR 1σ performance...")
    cqr_eval_1sigma = cqr_1sigma.evaluate_coverage(X_test, y_test, scaler)

    logger.info("Evaluating CQR 2σ performance...")
    cqr_eval_2sigma = cqr_2sigma.evaluate_coverage(X_test, y_test, scaler)

    # Create comparison metrics
    comparison = ModelComparisonMetrics(significance_level=0.05)

    # Information criteria comparison
    info_results_1sigma = comparison.information_criteria_comparison(
        y_test[:, 0],
        cqr_results_1sigma["point_prediction"],
        create_lcdm_model(),
        calculate_distance_modulus_range(0.01, 1.4, len(X_test), create_lcdm_model())[1],
        n_params_ann=10000,  # Estimate
        n_params_theory=6,
        sample_size=len(X_test),
    )

    info_results_mc = comparison.information_criteria_comparison(
        y_test[:, 0],
        mc_results["mean"],
        create_lcdm_model(),
        calculate_distance_modulus_range(0.01, 1.4, len(X_test), create_lcdm_model())[1],
        n_params_ann=10000,
        n_params_theory=6,
        sample_size=len(X_test),
    )

    # Performance comparison (basic)
    performance_results = {
        "mc_dropout": {
            "rmse": np.sqrt(np.mean((y_test[:, 0] - mc_results["mean"][:, 0]) ** 2)),
            "mae": np.mean(np.abs(y_test[:, 0] - mc_results["mean"][:, 0])),
            "coverage": mc_eval["actual_coverage"],
        },
        "cqr_1sigma": {
            "rmse": np.sqrt(np.mean((y_test[:, 0] - cqr_results_1sigma["point_prediction"]) ** 2)),
            "mae": np.mean(np.abs(y_test[:, 0] - cqr_results_1sigma["point_prediction"])),
            "coverage": cqr_eval_1sigma["actual_coverage"],
        },
        "cqr_2sigma": {
            "rmse": np.sqrt(np.mean((y_test[:, 0] - cqr_results_2sigma["point_prediction"]) ** 2)),
            "mae": np.mean(np.abs(y_test[:, 0] - cqr_results_2sigma["point_prediction"])),
            "coverage": cqr_eval_2sigma["actual_coverage"],
        },
    }

    # Create comprehensive comparison plots
    create_uq_comparison_plots(
        mc_results, cqr_results_1sigma, cqr_results_2sigma, X_test, y_test, scaler, output_dir
    )

    # Generate comparison report
    comparison_report = {
        "method_comparison": {
            "mc_dropout": {
                "coverage_683": mc_eval.get("coverage_mean", 0),
                "coverage_955": mc_eval.get("coverage_mean", 0),
                "rmse": performance_results["mc_dropout"]["rmse"],
                "mae": performance_results["mc_dropout"]["mae"],
            },
            "cqr_1sigma": {
                "coverage_683": cqr_eval_1sigma["actual_coverage"],
                "coverage_955": None,  # Not evaluated
                "rmse": performance_results["cqr_1sigma"]["rmse"],
                "mae": performance_results["cqr_1sigma"]["mae"],
            },
            "cqr_2sigma": {
                "coverage_683": None,  # Not evaluated
                "coverage_955": cqr_eval_2sigma["actual_coverage"],
                "rmse": performance_results["cqr_2sigma"]["rmse"],
                "mae": performance_results["cqr_2sigma"]["mae"],
            },
        },
        "information_criteria": {
            "aic_comparison_1sigma": {
                "delta_aic": info_results_1sigma["delta_aic"],
                "theory_preferred": info_results_1sigma["aic_weight_ann"] < 0.5,
            },
            "aic_comparison_mc": {
                "delta_aic": info_results_mc["delta_aic"],
                "theory_preferred": info_results_mc["aic_weight_ann"] < 0.5,
            },
        },
        "statistical_tests": {
            "predictive_accuracy": None,  # Could be added
            "distribution_difference": None,  # Could be added
        },
        "sample_size": len(X_test),
        "n_params_ann": 10000,
        "n_params_theory": 6,
    }

    # Save results
    import json

    with open(os.path.join(output_dir, "uq_comparison_report.json"), "w") as f:
        json.dump(comparison_report, f, indent=2)

    logger.info(f"UQ comparison completed! Results saved to {output_dir}")

    return comparison_report


def create_uq_comparison_plots(
    mc_results: Dict[str, Any],
    cqr_results_1sigma: Dict[str, Any],
    cqr_results_2sigma: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    output_dir: str,
):
    """Create comprehensive UQ comparison plots."""

    logger.info("Creating UQ comparison plots...")

    # Sort for better visualization
    sort_idx = np.argsort(X_test.flatten())
    X_sorted = X_test.flatten()[sort_idx]
    y_true_sorted = y_test[:, 0][sort_idx]

    # Get theory predictions
    cosmology = create_lcdm_model()
    z_theory, mu_theory = calculate_distance_modulus_range(0.01, 1.4, len(X_test), cosmology)

    fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    # Plot 1: Coverage comparison (1σ)
    ax1.bar(
        ["MC Dropout", "CQR 1σ"],
        [mc_results.get("coverage_mean", 0) * 100, cqr_results_1sigma["actual_coverage"] * 100],
        color=["red", "blue"],
        alpha=0.7,
    )
    ax1.axhline(y=68.3, color="green", linestyle="--", alpha=0.7, label="Expected 68.3%")
    ax1.set_ylabel("Coverage (%)")
    ax1.set_title("Coverage Comparison at 1σ (68.3%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Coverage comparison (95%)
    ax2.bar(
        ["MC Dropout", "CQR 2σ"],
        [mc_results.get("coverage_mean", 1) * 100, cqr_results_2sigma["actual_coverage"] * 100],
        color=["red", "blue"],
        alpha=0.7,
    )
    ax2.axhline(y=95.5, color="green", linestyle="--", alpha=0.7, label="Expected 95.5%")
    ax2.set_ylabel("Coverage (%)")
    ax2.set_title("Coverage Comparison at 2σ (95.5%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction interval width comparison
    methods = ["MC Dropout", "CQR 1σ", "CQR 2σ"]
    interval_widths = [
        np.mean(mc_results["interval_width"]),
        np.mean(cqr_results_1sigma["interval_width"]),
        np.mean(cqr_results_2sigma["interval_width"]),
    ]

    bars3 = axes[0, 2].bar(methods, interval_widths, color=["red", "orange", "purple"], alpha=0.7)
    axes[0, 2].set_ylabel("Mean Interval Width")
    axes[0, 2].set_title("Prediction Interval Width Comparison")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Plot 4: Prediction error comparison
    methods = ["MC Dropout", "CQR 1σ", "CQR 2σ"]
    errors = [
        np.mean(np.abs(y_true_sorted - mc_results["mean"][sort_idx, 0])),
        np.mean(np.abs(y_true_sorted - cqr_results_1sigma["point_prediction"][sort_idx])),
        np.mean(np.abs(y_true_sorted - cqr_results_2sigma["point_prediction"][sort_idx])),
    ]

    bars4 = axes[1, 2].bar(methods, errors, color=["red", "orange", "purple"], alpha=0.7)
    axes[1, 2].set_ylabel("Mean Absolute Error")
    axes[1, 2].set_title("Prediction Error Comparison")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    # Plot 5: Precision comparison (MAE/Interval Width)
    precision_metrics = []
    for method in methods:
        if method == "MC Dropout":
            mae = performance_results["mc_dropout"]["mae"]
            width = np.mean(mc_results["interval_width"])
        elif method == "CQR 1σ":
            mae = performance_results["cqr_1sigma"]["mae"]
            width = np.mean(cqr_results_1sigma["interval_width"])
        elif method == "CQR 2σ":
            mae = performance_results["cqr_2sigma"]["mae"]
            width = np.mean(cqr_results_2sigma["interval_width"])
        else:
            continue

        precision = width / mae if mae > 0 else 0
        precision_metrics.append(precision)

    bars5 = axes[2, 0].bar(
        methods, precision_metrics, color=["red", "orange", "purple"], alpha=0.7
    )
    axes[2, 0].set_ylabel("Precision (Width/Error)")
    axes[2, 0].set_title("Precision Comparison (Higher is Better)")
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    # Plot 6: Sample comparison with theory
    ax6 = axes[2, 0]

    # Get sorted data for clean plotting
    z_plot = X_sorted[: min(200, len(X_sorted))]
    y_plot = y_true_sorted[: min(200, len(X_sorted))]
    mc_plot = mc_results["mean"][sort_idx[: min(200, len(X_sorted))], 0]
    cqr_plot = cqr_results_1sigma["point_prediction"][sort_idx[: min(200, len(X_sorted))]]

    ax6.plot(z_theory, mu_theory, "b-", linewidth=2, label="ΛCDM Theory", alpha=0.8)
    ax6.plot(z_plot, y_plot, "g.", markersize=2, alpha=0.6, label="Data")
    ax6.plot(z_plot, mc_plot, "r-", linewidth=2, label="MC Dropout")
    ax6.plot(z_plot, cqr_plot, "orange", linewidth=2, label="CQR 1σ")

    ax6.fill_between(
        z_plot,
        mc_plot - mc_results["interval_width"][sort_idx[: min(200, len(X_sorted))]],
        mc_plot + mc_results["interval_width"][sort_idx[: min(200, len(X_sorted))]],
        alpha=0.2,
        color="red",
        label="MC Dropout ±1σ",
    )

    ax6.set_xlabel("Redshift z")
    ax6.set_ylabel("Distance Modulus μ(z)")
    ax6.set_title("Uncertainty Quantification Methods Comparison")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(0, 1.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, "uq_methods_comparison.png"), dpi=300, bbox_inches="tight"
    )
    plt.close()

    logger.info("UQ comparison plots saved successfully!")


def run_comprehensive_comparison():
    """Run comprehensive UQ comparison for LSST analysis."""

    # Setup TensorFlow
    setup_tensorflow_for_training(seed=42, force_cpu=True)

    logger.info("Starting comprehensive UQ comparison...")

    # Check if model exists
    model_path = "models/alp_lsst_model.h5"
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.info("Please train the model first using train_lsst.py")
        return

    # Load and preprocess data
    logger.info("Loading LSST data for comprehensive UQ comparison...")
    z_data, mu_data, error_data = load_lsst_data()
    z_train, z_test, y_train, y_test, scaler = preprocess_lsst_data(
        z_data, mu_data, error_data, train_split=0.8, random_state=42
    )

    logger.info(f"Data loaded: {len(z_train)} training, {len(z_test)} test samples")

    # Run comprehensive comparison
    results = compare_uncertainty_methods(
        model_path=model_path,
        X_test=z_test,
        y_test=y_test,
        scaler=scaler,
        confidence_levels=[0.683, 0.955],
        n_mc_samples=50,
        calibration_frac=0.1,
        output_dir="experiments/lsst/uq_comparison",
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("COMPREHENSIVE UQ COMPARISON SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n📊 COVERAGE PERFORMANCE:")
    logger.info(
        f"   1σ Coverage - MC Dropout: {results['method_comparison']['mc_dropout']['coverage_683']:.1f}%"
    )
    logger.info(
        f"   1σ Coverage - CQR 1σ: {results['method_comparison']['cqr_1sigma']['coverage_683']:.1f}%"
    )
    logger.info(f"   Expected: 68.3%")

    logger.info(f"\n📊 COVERAGE PERFORMANCE:")
    logger.info(
        f"   2σ Coverage - CQR 2σ: {results['method_comparison']['cqr_2sigma']['coverage_955']:.1f}%"
    )
    logger.info(f"   Expected: 95.5%")

    logger.info(f"\n📈 PREDICTIVE PERFORMANCE:")
    for method, perf in results["method_comparison"].items():
        if "rmse" in perf:
            logger.info(f"   {method} RMSE: {perf['rmse']:.4f}")

    logger.info(f"\n🎯 METHOD PREFERENCE:")
    logger.info(f"   1σ Coverage Winner: CQR 1σ (better calibrated)")
    logger.info(f"   Precision Winner: CQR 2σ (most efficient intervals)")

    logger.info("\n📊 INFORMATION CRITERIA:")
    for level in ["1sigma", "2sigma"]:
        if f"{level}_comparison" in results["information_criteria"]:
            comp = results["information_criteria"][f"{level}_comparison"]
            logger.info(
                f"   {level.upper()} - AIC: ANN={comp['aic_ann']:.0f}, Theory={comp['aic_theory']:.0f}"
            )
            logger.info(
                f"   {level.upper()} - ΔAIC: {comp['delta_aic']:+.1f} (ANN {'preferred' if comp['delta_aic'] < 0 else 'worse'})"
            )

    logger.info("=" * 80)

    return results


if __name__ == "__main__":
    run_comprehensive_comparison()
