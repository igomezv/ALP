#!/usr/bin/env python
"""Conformalized Quantile Regression for Uncertainty Quantification

This module implements Conformalized Quantile Regression (CQR) for distribution-free
prediction intervals with guaranteed coverage properties.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, List
from sklearn.preprocessing import StandardScaler

from ..utils.logger_config import logger


def quantile_loss(y_true, y_pred, q):
    """Quantile loss function for a given quantile q."""
    error = y_true - y_pred
    return tf.reduce_mean(tf.maximum(q * error, (q - 1) * error))


class QuantileRegressionNN(tf.keras.Model):
    """Neural network for quantile regression.

    Implements quantile loss for predicting specified quantiles.
    """

    def __init__(self, quantile: float, hidden_layers=[200, 200, 200], dropout_rate=0.1, **kwargs):
        super(QuantileRegressionNN, self).__init__(**kwargs)
        self.quantile = quantile
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate

        # Build layers
        self.dense_layers = []
        self.dropout_layers = []

        for units in hidden_layers:
            self.dense_layers.append(tf.keras.layers.Dense(units, activation="relu"))
            self.dropout_layers.append(tf.keras.layers.Dropout(dropout_rate))

        self.output_layer = tf.keras.layers.Dense(1)

    def call(self, inputs, training=None):
        """Forward pass with quantile loss."""
        x = inputs

        for dense, dropout in zip(self.dense_layers, self.dropout_layers):
            x = dense(x)
            x = dropout(x, training=training)

        return self.output_layer(x)

    def get_quantile_loss(self):
        """Get quantile loss function for this model's quantile."""

        def loss(y_true, y_pred):
            return quantile_loss(y_true, y_pred, self.quantile)

        return loss


class ConformalizedQuantileRegression:
    """Conformalized Quantile Regression for adaptive prediction intervals.

    Combines quantile regression with conformal prediction to create
    adaptive prediction intervals with guaranteed coverage.
    """

    def __init__(
        self,
        lower_quantile: float = 0.1587,  # -1σ (68.3% CI)
        upper_quantile: float = 0.8413,  # +1σ (68.3% CI)
        calibration_frac: float = 0.1,
    ):
        """Initialize CQR with specified confidence level.

        Args:
            lower_quantile: Lower quantile for prediction interval
            upper_quantile: Upper quantile for prediction interval
            calibration_frac: Fraction of training data for calibration
        """
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.calibration_frac = calibration_frac
        self.confidence_level = upper_quantile - lower_quantile

        self.lower_model = None
        self.upper_model = None
        self.median_model = None  # For point predictions
        self.calibration_scores = None

        logger.info(f"CQR initialized with {self.confidence_level:.1%} confidence interval")
        logger.info(f"Using {calibration_frac:.1%} of data for calibration")

    def create_quantile_model(self, quantile: float = None) -> QuantileRegressionNN:
        """Create quantile regression neural network."""

        if quantile is None:
            quantile = self.lower_quantile

        model = QuantileRegressionNN(quantile=quantile)
        model.compile(optimizer="adam", loss=model.get_quantile_loss(), metrics=["mae"])

        return model

    def train_quantile_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        validation_split: float = 0.2,
        epochs: int = 200,
        batch_size: int = 32,
        verbose: int = 0,
    ) -> Dict[str, Any]:
        """Train lower, upper, and median quantile models."""

        logger.info("Training quantile regression models for CQR...")

        # Preprocess input data
        if scaler is not None:
            X_processed = scaler.transform(X_train.reshape(-1, 1))
        elif len(X_train.shape) == 1:
            X_processed = X_train.reshape(-1, 1)
        else:
            X_processed = X_train

        # Split data: calibration set + training set + validation set
        n_calib = int(len(X_processed) * self.calibration_frac)
        n_total = len(X_processed)

        # Use first portion for calibration (to ensure distribution matching)
        X_calib = X_processed[:n_calib]
        y_calib = y_train[:n_calib]
        X_model_train = X_processed[n_calib:]
        y_model_train = y_train[n_calib:]

        logger.info(f"Calibration set: {len(X_calib)} samples")
        logger.info(f"Training set: {len(X_model_train)} samples")

        # Create models
        self.lower_model = self.create_quantile_model(quantile=self.lower_quantile)
        self.upper_model = self.create_quantile_model(quantile=self.upper_quantile)
        self.median_model = self.create_quantile_model(quantile=0.5)

        # Train models
        models_data = [
            (self.lower_model, "Lower Quantile"),
            (self.upper_model, "Upper Quantile"),
            (self.median_model, "Median"),
        ]

        training_history = {}

        for model, name in models_data:
            logger.info(f"Training {name} model...")

            # Early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=20, restore_best_weights=True, verbose=0
            )

            history = model.fit(
                X_model_train,
                y_model_train,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                callbacks=[early_stopping],
            )

            training_history[name] = history

        # Compute calibration scores using conformal prediction
        self._compute_calibration_scores(X_calib, y_calib, scaler)

        logger.info("CQR training completed successfully")

        return {
            "training_history": training_history,
            "calibration_scores": self.calibration_scores,
            "n_calibration": len(X_calib),
        }

    def _compute_calibration_scores(
        self, X_calib: np.ndarray, y_calib: np.ndarray, scaler: Optional[StandardScaler] = None
    ):
        """Compute calibration scores for conformal prediction."""

        logger.info("Computing calibration scores...")

        # Get quantile predictions on calibration set
        lower_pred = self.lower_model.predict(X_calib, verbose=0).flatten()
        upper_pred = self.upper_model.predict(X_calib, verbose=0).flatten()
        median_pred = self.median_model.predict(X_calib, verbose=0).flatten()

        # Compute non-conformity scores
        # For CQR: max(lower - y, y - upper)
        y_calib_flat = y_calib.flatten() if y_calib.ndim > 1 else y_calib

        # Ensure compatible shapes
        if len(lower_pred) != len(y_calib_flat):
            y_calib_flat = y_calib_flat[: len(lower_pred)]
        if len(upper_pred) != len(y_calib_flat):
            y_calib_flat = y_calib_flat[: len(upper_pred)]

        lower_scores = np.maximum(lower_pred - y_calib_flat, 0)
        upper_scores = np.maximum(y_calib_flat - upper_pred, 0)
        calibration_scores = np.maximum(lower_scores, upper_scores)

        # Store for use in prediction
        self.calibration_scores = calibration_scores

        logger.info(
            f"Calibration scores computed: mean={np.mean(calibration_scores):.4f}, std={np.std(calibration_scores):.4f}"
        )

    def predict_with_intervals(
        self,
        X_test: np.ndarray,
        scaler: Optional[StandardScaler] = None,
        alpha: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Make predictions with conformalized prediction intervals."""

        if self.calibration_scores is None:
            raise ValueError("Model not trained. Call train_quantile_models first.")

        if alpha is None:
            alpha = 1 - self.confidence_level

        # Preprocess input data
        if scaler is not None:
            X_processed = scaler.transform(X_test.reshape(-1, 1))
        elif len(X_test.shape) == 1:
            X_processed = X_test.reshape(-1, 1)
        else:
            X_processed = X_test

        # Get quantile predictions
        lower_pred = self.lower_model.predict(X_processed, verbose=0).flatten()
        upper_pred = self.upper_model.predict(X_processed, verbose=0).flatten()
        median_pred = self.median_model.predict(X_processed, verbose=0).flatten()

        # Compute conformal correction
        # Use (1 + 1/n_calib) quantile of calibration scores
        n_calib = len(self.calibration_scores)
        correction = np.quantile(self.calibration_scores, 1 - alpha)

        # Apply correction to create valid prediction intervals
        lower_conformal = lower_pred - correction
        upper_conformal = upper_pred + correction

        # Point prediction (median)
        point_prediction = median_pred

        # Uncertainty estimate (half interval width)
        uncertainty = (upper_conformal - lower_conformal) / 2

        results = {
            "point_prediction": point_prediction,
            "lower_bound": lower_conformal,
            "upper_bound": upper_conformal,
            "uncertainty": uncertainty,
            "quantile_uncertainty": (upper_pred - lower_pred) / 2,  # Before conformalization
            "conformal_correction": correction,
            "interval_width": upper_conformal - lower_conformal,
        }

        logger.info(f"CQR predictions computed for {len(X_test)} samples")
        logger.info(
            f"Mean correction: {correction:.4f}, Mean interval width: {np.mean(results['interval_width']):.4f}"
        )

        return results

    def evaluate_coverage(
        self, X_test: np.ndarray, y_true: np.ndarray, scaler: Optional[StandardScaler] = None
    ) -> Dict[str, float]:
        """Evaluate prediction interval coverage on test data."""

        results = self.predict_with_intervals(X_test, scaler)

        # Check coverage
        y_true_flat = y_true.flatten()
        lower_bound = results["lower_bound"]
        upper_bound = results["upper_bound"]

        # Ensure compatible shapes for coverage calculation
        min_len = min(len(y_true_flat), len(lower_bound), len(upper_bound))
        y_true_trim = y_true_flat[:min_len]
        lower_trim = lower_bound[:min_len]
        upper_trim = upper_bound[:min_len]
        point_pred_trim = results["point_prediction"][:min_len]

        within_interval = (y_true_trim >= lower_trim) & (y_true_trim <= upper_trim)
        actual_coverage = np.mean(within_interval)
        expected_coverage = self.confidence_level

        # Additional metrics
        mean_width = np.mean(results["interval_width"][:min_len])
        mae = np.mean(np.abs(y_true_trim - point_pred_trim))

        coverage_metrics = {
            "actual_coverage": actual_coverage,
            "expected_coverage": expected_coverage,
            "coverage_difference": actual_coverage - expected_coverage,
            "mean_interval_width": mean_width,
            "mae": mae,
            "coverage_efficiency": (1 - mean_width) * actual_coverage,  # Higher is better
        }

        logger.info(f"CQR Coverage: {actual_coverage:.3f} (expected: {expected_coverage:.3f})")
        logger.info(f"Mean interval width: {mean_width:.4f}, MAE: {mae:.4f}")

        return coverage_metrics

    def save_models(self, filepath_prefix: str):
        """Save trained quantile models."""
        if self.lower_model is None:
            raise ValueError("No models to save. Train models first.")

        self.lower_model.save(f"{filepath_prefix}_lower.h5")
        self.upper_model.save(f"{filepath_prefix}_upper.h5")
        self.median_model.save(f"{filepath_prefix}_median.h5")

        # Save calibration scores
        np.save(f"{filepath_prefix}_calibration_scores.npy", self.calibration_scores)

        logger.info(f"CQR models saved to {filepath_prefix}_*.h5")

    def load_models(self, filepath_prefix: str):
        """Load trained quantile models."""
        self.lower_model = tf.keras.models.load_model(f"{filepath_prefix}_lower.h5")
        self.upper_model = tf.keras.models.load_model(f"{filepath_prefix}_upper.h5")
        self.median_model = tf.keras.models.load_model(f"{filepath_prefix}_median.h5")

        try:
            self.calibration_scores = np.load(f"{filepath_prefix}_calibration_scores.npy")
        except FileNotFoundError:
            logger.warning("Calibration scores not found. May need retraining.")

        logger.info(f"CQR models loaded from {filepath_prefix}_*.h5")


def create_cqr_for_sigma(
    sigma: float, calibration_frac: float = 0.1
) -> ConformalizedQuantileRegression:
    """Convenience function to create CQR for specific sigma levels."""

    from scipy import stats

    # Convert sigma to quantiles
    lower_quantile = stats.norm.cdf(-sigma)
    upper_quantile = stats.norm.cdf(sigma)

    return ConformalizedQuantileRegression(
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
        calibration_frac=calibration_frac,
    )
