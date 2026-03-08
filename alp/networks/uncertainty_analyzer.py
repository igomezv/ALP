#!/usr/bin/env python

"""
Astro Layer Perceptron Networks Module - Uncertainty Analysis
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
Comprehensive uncertainty analysis and error evaluation tools
for neural network predictions with MC Dropout uncertainty quantification.
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path

from .uncertainty import UncertaintyQuantifier
from ..utils.logger_config import logger


class UncertaintyAnalyzer:
    """
    Comprehensive uncertainty analysis for neural network predictions.

    Provides tools for analyzing MC Dropout predictions, evaluating uncertainty
    quality, calibrating confidence intervals, and visualizing uncertainty metrics.

    Parameters
    ----------
    n_samples : int, optional
        Number of MC Dropout samples for uncertainty estimation (default 100)
    confidence_levels : list, optional
        Confidence levels for interval analysis (default [0.68, 0.95, 0.99])
    """

    def __init__(self, n_samples: int = 100, confidence_levels: List[float] = [0.68, 0.95, 0.99]):
        self.n_samples = n_samples
        self.confidence_levels = confidence_levels
        self.uq = UncertaintyQuantifier(n_samples=n_samples)

        # Storage for analysis results
        self.predictions = None
        self.uncertainties = None
        self.analysis_results = {}

    def analyze_predictions(
        self, model: tf.keras.Model, X_test: np.ndarray, y_true: np.ndarray, scaler=None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive uncertainty analysis on model predictions.

        Parameters
        ----------
        model : tf.keras.Model
            Trained model with MC Dropout layers
        X_test : np.ndarray
            Test input data
        y_true : np.ndarray
            True target values
        scaler : sklearn scaler, optional
            Data scaler for inverse transformation

        Returns
        -------
        dict
            Comprehensive analysis results
        """
        logger.info(f"Starting uncertainty analysis with {self.n_samples} MC samples...")

        # Generate MC Dropout predictions and store all predictions
        # First generate all predictions manually to store them
        logger.info(f"Performing {self.n_samples} MC dropout predictions...")
        all_predictions = []

        # Prepare input data
        if scaler is not None:
            X_processed = scaler.transform(X_test.reshape(-1, 1))
        elif len(X_test.shape) == 1:
            X_processed = X_test.reshape(-1, 1)
        else:
            X_processed = X_test

        for i in range(self.n_samples):
            if (i + 1) % 20 == 0:
                logger.info(f"  MC prediction {i + 1}/{self.n_samples}")
            pred = model(X_processed, training=True).numpy()
            all_predictions.append(pred)

        all_predictions = np.array(all_predictions)
        mean_pred = np.mean(all_predictions, axis=0)
        std_pred = np.std(all_predictions, axis=0, ddof=1)

        # Calculate combined uncertainty for Y1 (distance modulus)
        combined_uncertainty = np.sqrt(
            std_pred[:, 0] ** 2 + std_pred[:, 1] ** 2 + mean_pred[:, 1] ** 2
        )

        # Store model and data for later use
        self._model = model
        self._X_analysis = X_test
        self._scaler = scaler

        # Create mc_results dict with all predictions
        mc_results = {
            "mean": mean_pred,
            "std": std_pred,
            "combined_uncertainty": combined_uncertainty,
            "all_predictions": all_predictions,
        }

        # Extract predictions and uncertainties
        self.predictions = mc_results["mean"]

        # Calculate epistemic and aleatoric components
        # Epistemic: MC dropout std for primary output
        epistemic_uncertainty = std_pred[:, 0]

        # Aleatoric: Inferred from second output and combined uncertainty
        # combined_uncertainty = sqrt(epistemic^2 + aleatoric^2)
        aleatoric_uncertainty = np.sqrt(
            np.maximum(0, combined_uncertainty**2 - epistemic_uncertainty**2)
        )

        self.uncertainties = {
            "epistemic": epistemic_uncertainty,
            "aleatoric": aleatoric_uncertainty,
            "combined": combined_uncertainty,
            "total_std": std_pred,  # Keep original std for reference
        }

        # Perform analysis
        self.analysis_results = {
            "predictions": self.predictions,
            "uncertainties": self.uncertainties,
            "error_metrics": self._compute_error_metrics(y_true, self.predictions),
            "uncertainty_calibration": self._analyze_uncertainty_calibration(
                y_true, self.predictions, self.uncertainties
            ),
            "confidence_intervals": self._compute_confidence_intervals(mc_results),
            "reliability_analysis": self._analyze_reliability(
                y_true, self.predictions, self.uncertainties
            ),
            "uncertainty_decomposition": self._analyze_uncertainty_decomposition(),
        }

        logger.info("Uncertainty analysis completed!")
        return self.analysis_results

    def _compute_error_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Compute standard error metrics."""
        metrics = {}

        # Handle multi-dimensional outputs
        if y_true.ndim > 1:
            # Compute metrics for each output dimension
            for i in range(y_true.shape[1]):
                metrics[f"rmse_output_{i}"] = np.sqrt(
                    mean_squared_error(y_true[:, i], y_pred[:, i])
                )
                metrics[f"mae_output_{i}"] = mean_absolute_error(y_true[:, i], y_pred[:, i])
                metrics[f"r2_output_{i}"] = r2_score(y_true[:, i], y_pred[:, i])

            # Overall metrics (flatten)
            metrics["rmse_overall"] = np.sqrt(
                mean_squared_error(y_true.flatten(), y_pred.flatten())
            )
            metrics["mae_overall"] = mean_absolute_error(y_true.flatten(), y_pred.flatten())
            metrics["r2_overall"] = r2_score(y_true.flatten(), y_pred.flatten())
        else:
            metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
            metrics["mae"] = mean_absolute_error(y_true, y_pred)
            metrics["r2"] = r2_score(y_true, y_pred)

        # Normalized error metrics
        errors = np.abs(y_true - y_pred)
        metrics["mean_error"] = np.mean(errors)
        metrics["std_error"] = np.std(errors)
        metrics["max_error"] = np.max(errors)

        return metrics

    def _analyze_uncertainty_calibration(
        self, y_true: np.ndarray, y_pred: np.ndarray, uncertainties: Dict[str, np.ndarray]
    ) -> Dict[str, Any]:
        """Analyze uncertainty calibration and quality."""
        calibration = {}

        # Compute prediction errors
        errors = np.abs(y_true - y_pred)

        for unc_type, unc_values in uncertainties.items():
            if unc_type == "aleatoric" and np.all(unc_values == 0):
                continue  # Skip aleatoric if not computed
            if unc_type == "total_std":
                continue  # Skip total std, use specific uncertainty types

            # Calibration metrics
            calibration[unc_type] = {}

            # Use first output for calibration (assuming y_true is (N, 2) or (N,))
            if errors.ndim > 1:
                errors_flat = errors[:, 0]  # Use first output for calibration
            else:
                errors_flat = errors

            # Mean absolute error vs mean uncertainty
            mae_vs_unc = np.mean(errors_flat) / np.mean(unc_values)
            calibration[unc_type]["mae_to_uncertainty_ratio"] = mae_vs_unc

            # Correlation between error and uncertainty
            corr = np.corrcoef(errors_flat, unc_values)[0, 1] if len(errors_flat) > 1 else 0
            calibration[unc_type]["error_uncertainty_correlation"] = corr

            # Coverage analysis for different confidence levels
            coverage = {}
            for level in self.confidence_levels:
                z_score = stats.norm.ppf((1 + level) / 2)
                within_interval = errors_flat <= z_score * unc_values
                coverage[f"coverage_{level}"] = np.mean(within_interval)
                coverage[f"expected_coverage_{level}"] = level
                coverage[f"coverage_diff_{level}"] = coverage[f"coverage_{level}"] - level

            calibration[unc_type]["coverage_analysis"] = coverage

        return calibration

    def _compute_confidence_intervals(
        self, mc_results: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Compute confidence intervals from MC predictions."""
        intervals = {}

        for level in self.confidence_levels:
            alpha = 1 - level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            intervals[f"ci_{level}_lower"] = np.percentile(
                mc_results["all_predictions"], lower_percentile, axis=0
            )
            intervals[f"ci_{level}_upper"] = np.percentile(
                mc_results["all_predictions"], upper_percentile, axis=0
            )
            intervals[f"ci_{level}_width"] = (
                intervals[f"ci_{level}_upper"] - intervals[f"ci_{level}_lower"]
            )

        return intervals

    def _analyze_reliability(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        uncertainties: Dict[str, np.ndarray],
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Analyze reliability of uncertainty estimates."""
        reliability = {}

        for unc_type, unc_values in uncertainties.items():
            if unc_type == "aleatoric" and np.all(unc_values == 0):
                continue

            # Bin predictions by uncertainty magnitude
            errors = np.abs(y_true - y_pred)
            unc_flat = unc_values.flatten()
            err_flat = errors.flatten()

            # Sort by uncertainty
            sorted_indices = np.argsort(unc_flat)
            n_samples = len(unc_flat)
            bin_size = n_samples // n_bins

            bin_uncertainties = []
            bin_errors = []

            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = (i + 1) * bin_size if i < n_bins - 1 else n_samples

                bin_unc = unc_flat[sorted_indices[start_idx:end_idx]]
                bin_err = err_flat[sorted_indices[start_idx:end_idx]]

                bin_uncertainties.append(np.mean(bin_unc))
                bin_errors.append(np.mean(bin_err))

            reliability[unc_type] = {
                "uncertainty_bins": bin_uncertainties,
                "error_bins": bin_errors,
                "reliability_correlation": np.corrcoef(bin_uncertainties, bin_errors)[0, 1]
                if len(bin_uncertainties) > 1
                else 0,
            }

        return reliability

    def _analyze_uncertainty_decomposition(self) -> Dict[str, Any]:
        """Analyze the decomposition of uncertainty sources."""
        if "epistemic" not in self.uncertainties or "aleatoric" not in self.uncertainties:
            return {"decomposition": "Not available - missing uncertainty components"}

        epistemic = self.uncertainties["epistemic"]
        aleatoric = self.uncertainties["aleatoric"]
        combined = self.uncertainties["combined"]

        decomposition = {
            "mean_epistemic": np.mean(epistemic),
            "mean_aleatoric": np.mean(aleatoric),
            "mean_combined": np.mean(combined),
            "epistemic_fraction": np.mean(epistemic) / np.mean(combined)
            if np.mean(combined) > 0
            else 0,
            "aleatoric_fraction": np.mean(aleatoric) / np.mean(combined)
            if np.mean(combined) > 0
            else 0,
        }

        # Spatial analysis of uncertainty contributions
        epistemic_ratio = epistemic / (combined + 1e-8)
        decomposition["epistemic_dominance_fraction"] = np.mean(epistemic_ratio > 0.5)
        decomposition["aleatoric_dominance_fraction"] = np.mean(epistemic_ratio <= 0.5)

        return {"decomposition": decomposition}

    def plot_uncertainty_analysis(
        self,
        X: np.ndarray,
        y_true: np.ndarray,
        output_dir: str = "outputs",
        prefix: str = "uncertainty",
    ):
        """Generate comprehensive uncertainty analysis plots."""
        if not self.analysis_results:
            logger.error("No analysis results available. Run analyze_predictions first.")
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Prediction vs True with uncertainty bands
        self._plot_prediction_uncertainty(X, y_true, output_dir, prefix)

        # 2. Error vs Uncertainty calibration
        self._plot_error_uncertainty_calibration(y_true, output_dir, prefix)

        # 3. Confidence interval coverage
        self._plot_coverage_analysis(y_true, output_dir, prefix)

        # 4. Reliability diagram
        self._plot_reliability_diagram(output_dir, prefix)

        # 5. Uncertainty decomposition
        self._plot_uncertainty_decomposition(output_dir, prefix)

        logger.info(f"Uncertainty analysis plots saved to {output_dir}/")

    def _plot_prediction_uncertainty(
        self, X: np.ndarray, y_true: np.ndarray, output_dir: str, prefix: str
    ):
        """Plot predictions with uncertainty bands."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Handle multi-dimensional outputs
        if y_true.ndim > 1:
            n_outputs = y_true.shape[1]
        else:
            n_outputs = 1
            y_true = y_true.reshape(-1, 1)
            self.predictions = self.predictions.reshape(-1, 1)

        for i in range(min(n_outputs, 4)):  # Plot max 4 outputs
            ax = axes[i // 2, i % 2] if n_outputs > 1 else axes

            # Sort by X for better visualization
            if X.ndim == 1:
                sort_idx = np.argsort(X.flatten())
                X_sorted = X.flatten()[sort_idx]
                y_true_sorted = y_true[:, i][sort_idx]
                pred_sorted = self.predictions[:, i][sort_idx]
                unc_sorted = self.uncertainties["combined"].flatten()[sort_idx]
            else:
                # Use first dimension if multi-dim input
                X_sorted = np.arange(len(y_true))
                y_true_sorted = y_true[:, i]
                pred_sorted = self.predictions[:, i]
                unc_sorted = self.uncertainties["combined"].flatten()

            # Plot prediction with uncertainty band
            ax.plot(X_sorted, y_true_sorted, "b.", label="True", alpha=0.6, markersize=4)
            ax.plot(X_sorted, pred_sorted, "r-", label="Prediction", linewidth=2)
            ax.fill_between(
                X_sorted,
                pred_sorted - unc_sorted,
                pred_sorted + unc_sorted,
                alpha=0.3,
                color="red",
                label="±1σ Uncertainty",
            )

            ax.set_xlabel("Input" if X.ndim == 1 else "Sample Index")
            ax.set_ylabel(f"Output {i + 1}")
            ax.set_title(f"Prediction with Uncertainty - Output {i + 1}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        if n_outputs < 4:
            for i in range(n_outputs, 4):
                axes[i // 2, i % 2].set_visible(False)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_predictions.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _plot_error_uncertainty_calibration(
        self, y_true: np.ndarray, output_dir: str, prefix: str
    ):
        """Plot error vs uncertainty calibration."""
        fig, axes = plt.subplots(
            1, len(self.uncertainties), figsize=(5 * len(self.uncertainties), 5)
        )

        if len(self.uncertainties) == 1:
            axes = [axes]

        for i, (unc_type, unc_values) in enumerate(self.uncertainties.items()):
            if unc_type == "aleatoric" and np.all(unc_values == 0):
                axes[i].text(
                    0.5,
                    0.5,
                    f"{unc_type} not available",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                continue

            # Use first output for error calculation
            if y_true.ndim > 1:
                errors_flat = np.abs(y_true[:, 0] - self.predictions[:, 0])
            else:
                errors_flat = np.abs(
                    y_true - self.predictions[:, 0]
                    if self.predictions.ndim > 1
                    else self.predictions
                )

            # Debug prints to understand the shapes
            print(
                f"Debug - {unc_type}: unc_values.shape={getattr(unc_values, 'shape', len(unc_values))}, errors_flat.shape={errors_flat.shape}"
            )
            print(
                f"Debug - {unc_type}: len(unc_values)={len(unc_values)}, len(errors_flat)={len(errors_flat)}"
            )

            # Handle case where uncertainty is multidimensional
            if unc_values.ndim > 1:
                # Use the first output's uncertainty for plotting
                unc_plot = unc_values[: len(errors_flat), 0]
                errors_plot = errors_flat
            else:
                # Ensure both arrays have same length
                min_len = min(len(unc_values), len(errors_flat))
                unc_plot = unc_values[:min_len]
                errors_plot = errors_flat[:min_len]

            # For 1D case, also truncate errors_flat if needed
            if unc_values.ndim == 1:
                min_len = min(len(unc_values), len(errors_flat))
                unc_plot = unc_values[:min_len]
                errors_plot = errors_flat[:min_len]

            # Scatter plot with density
            axes[i].scatter(unc_plot, errors_plot, alpha=0.5, s=1)

            # Add diagonal line (perfect calibration)
            max_val = max(np.max(unc_plot), np.max(errors_plot))
            axes[i].plot([0, max_val], [0, max_val], "r--", label="Perfect Calibration")

            # Add regression line
            z_fit = np.polyfit(unc_plot, errors_plot, 1)
            p = np.poly1d(z_fit)
            x_line = np.linspace(0, max_val, 100)
            axes[i].plot(x_line, p(x_line), "g-", label=f"Linear Fit (slope={z_fit[0]:.2f})")

            axes[i].set_xlabel(f"{unc_type.replace('_', ' ').title()}")
            axes[i].set_ylabel("Absolute Error")
            axes[i].set_title(f"Error vs {unc_type.replace('_', ' ').title()} Calibration")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_calibration.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _plot_coverage_analysis(self, y_true: np.ndarray, output_dir: str, prefix: str):
        """Plot confidence interval coverage analysis."""
        if "confidence_intervals" not in self.analysis_results:
            return

        intervals = self.analysis_results["confidence_intervals"]
        calibration = self.analysis_results["uncertainty_calibration"]

        fig, ax = plt.subplots(figsize=(10, 6))

        expected_coverage = []
        actual_coverage = []

        for level in self.confidence_levels:
            expected_coverage.append(level)
            # Calculate actual coverage
            if "combined" in calibration:
                actual_coverage.append(
                    calibration["combined"]["coverage_analysis"].get(f"coverage_{level}", 0)
                )
            else:
                actual_coverage.append(0)

        # Plot coverage comparison
        x = np.arange(len(self.confidence_levels))
        width = 0.35

        ax.bar(x - width / 2, expected_coverage, width, label="Expected Coverage", alpha=0.7)
        ax.bar(x + width / 2, actual_coverage, width, label="Actual Coverage", alpha=0.7)

        ax.set_xlabel("Confidence Level")
        ax.set_ylabel("Coverage Probability")
        ax.set_title("Confidence Interval Coverage Analysis")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(l * 100)}%" for l in self.confidence_levels])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add coverage values on bars
        for i, (exp, act) in enumerate(zip(expected_coverage, actual_coverage)):
            ax.text(i - width / 2, exp + 0.01, f"{exp:.2f}", ha="center", va="bottom")
            ax.text(i + width / 2, act + 0.01, f"{act:.2f}", ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_coverage.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _plot_reliability_diagram(self, output_dir: str, prefix: str):
        """Plot reliability diagram for uncertainty estimates."""
        if "reliability_analysis" not in self.analysis_results:
            return

        reliability = self.analysis_results["reliability_analysis"]

        fig, axes = plt.subplots(1, len(reliability), figsize=(5 * len(reliability), 5))

        if len(reliability) == 1:
            axes = [axes]

        for i, (unc_type, rel_data) in enumerate(reliability.items()):
            if "uncertainty_bins" not in rel_data:
                axes[i].text(
                    0.5,
                    0.5,
                    f"{unc_type} not available",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                continue

            unc_bins = rel_data["uncertainty_bins"]
            err_bins = rel_data["error_bins"]

            # Plot reliability curve
            axes[i].plot(unc_bins, err_bins, "bo-", label="Reliability Curve", linewidth=2)

            # Plot perfect calibration line
            max_val = max(max(unc_bins), max(err_bins))
            axes[i].plot(
                [0, max_val], [0, max_val], "r--", label="Perfect Calibration", linewidth=2
            )

            axes[i].set_xlabel(f"Mean {unc_type.replace('_', ' ').title()}")
            axes[i].set_ylabel("Mean Absolute Error")
            axes[i].set_title(f"Reliability Diagram - {unc_type.replace('_', ' ').title()}")
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_reliability.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    def _plot_uncertainty_decomposition(self, output_dir: str, prefix: str):
        """Plot uncertainty decomposition analysis."""
        if "uncertainty_decomposition" not in self.analysis_results:
            return

        decomp = self.analysis_results["uncertainty_decomposition"]
        if "decomposition" not in decomp:
            return

        data = decomp["decomposition"]

        if isinstance(data, str):  # Error message
            logger.info(data)
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Pie chart of uncertainty contributions
        labels = ["Epistemic", "Aleatoric"]
        sizes = [data["epistemic_fraction"], data["aleatoric_fraction"]]
        colors = ["lightcoral", "lightblue"]

        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Uncertainty Decomposition")

        # Bar plot of uncertainty magnitudes
        unc_types = ["Epistemic", "Aleatoric", "Combined"]
        unc_values = [data["mean_epistemic"], data["mean_aleatoric"], data["mean_combined"]]

        bars = ax2.bar(unc_types, unc_values, color=["lightcoral", "lightblue", "lightgreen"])
        ax2.set_ylabel("Mean Uncertainty")
        ax2.set_title("Uncertainty Magnitudes")
        ax2.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, unc_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(unc_values) * 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"{prefix}_decomposition.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    def save_analysis_report(
        self, output_dir: str = "outputs", filename: str = "uncertainty_analysis_report.json"
    ):
        """Save comprehensive analysis report to JSON file."""
        if not self.analysis_results:
            logger.error("No analysis results available. Run analyze_predictions first.")
            return

        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.int64) or isinstance(obj, np.int32):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj

        serializable_results = convert_numpy(self.analysis_results)

        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, filename), "w") as f:
            import json

            json.dump(serializable_results, f, indent=2)

        logger.info(f"Analysis report saved to {output_dir}/{filename}")


def load_and_analyze_model(
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_samples: int = 100,
    scaler=None,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """
    Convenience function to load a model and perform uncertainty analysis.

    Parameters
    ----------
    model_path : str
        Path to the saved .h5 model file
    X_test : np.ndarray
        Test input data
    y_test : np.ndarray
        Test target data
    n_samples : int, optional
        Number of MC Dropout samples (default 100)
    scaler : sklearn scaler, optional
        Data scaler for inverse transformation
    output_dir : str, optional
        Directory to save results (default 'outputs')

    Returns
    -------
    dict
        Analysis results
    """
    # Load model
    from .base_networks import SupervisedNET

    base_net = SupervisedNET()
    model = base_net.load_model(model_path)

    # Create analyzer and run analysis
    analyzer = UncertaintyAnalyzer(n_samples=n_samples)
    results = analyzer.analyze_predictions(model, X_test, y_test, scaler=scaler)

    # Generate plots and save report
    analyzer.plot_uncertainty_analysis(X_test, y_test, output_dir)
    analyzer.save_analysis_report(output_dir)

    return results
