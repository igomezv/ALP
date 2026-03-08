#!/usr/bin/env python
"""Rigorous Model Comparison Methods for ANN vs ΛCDM

This module implements statistical and information-theoretic methods for comparing
neural network predictions with theoretical cosmological models.
"""

import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Optional, Tuple

from ..utils.logger_config import logger


class ModelComparisonMetrics:
    """Comprehensive model comparison metrics for ANN vs theoretical models."""

    def __init__(self, significance_level: float = 0.05):
        """Initialize comparison metrics.

        Args:
            significance_level: Statistical significance level (default 0.05)
        """
        self.significance_level = significance_level
        logger.info(
            f"ModelComparisonMetrics initialized with significance level {significance_level}"
        )

    def information_criteria_comparison(
        self,
        y_true: np.ndarray,
        y_pred_ann: np.ndarray,
        y_pred_theory: np.ndarray,
        n_params_ann: int,
        n_params_theory: int = 6,  # Typical LCDM parameters
        sample_size: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute information criteria (AIC, BIC, WAIC) for model comparison.

        Args:
            y_true: True values
            y_pred_ann: Neural network predictions
            y_pred_theory: Theoretical model predictions
            n_params_ann: Number of parameters in ANN
            n_params_theory: Number of parameters in theoretical model
            sample_size: Sample size (if None, uses len(y_true))

        Returns:
            Dictionary with information criteria metrics
        """
        if sample_size is None:
            sample_size = len(y_true)

        logger.info("Computing information criteria...")

        # Compute log-likelihoods (assuming Gaussian noise)
        sigma_ann = np.std(y_true - y_pred_ann)
        sigma_theory = np.std(y_true - y_pred_theory)

        # Avoid log(0)
        sigma_ann = max(sigma_ann, 1e-8)
        sigma_theory = max(sigma_theory, 1e-8)

        # Log-likelihoods
        log_likelihood_ann = -0.5 * sample_size * np.log(2 * np.pi * sigma_ann**2) - 0.5 * np.sum(
            ((y_true - y_pred_ann) ** 2) / sigma_ann**2
        )

        log_likelihood_theory = -0.5 * sample_size * np.log(
            2 * np.pi * sigma_theory**2
        ) - 0.5 * np.sum(((y_true - y_pred_theory) ** 2) / sigma_theory**2)

        # AIC (Akaike Information Criterion)
        aic_ann = 2 * n_params_ann - 2 * log_likelihood_ann
        aic_theory = 2 * n_params_theory - 2 * log_likelihood_theory

        # BIC (Bayesian Information Criterion)
        bic_ann = n_params_ann * np.log(sample_size) - 2 * log_likelihood_ann
        bic_theory = n_params_theory * np.log(sample_size) - 2 * log_likelihood_theory

        # WAIC (Watanabe-Akaike Information Criterion) approximation
        # Using simpler version: AIC + 2k for effective parameters
        k_eff_ann = min(n_params_ann, sample_size / 10)  # Rough approximation
        k_eff_theory = min(n_params_theory, sample_size / 10)

        waic_ann = aic_ann + 2 * k_eff_ann
        waic_theory = aic_theory + 2 * k_eff_theory

        # Model preference (lower is better)
        delta_aic = aic_ann - aic_theory
        delta_bic = bic_ann - bic_theory
        delta_waic = waic_ann - waic_theory

        # Evidence ratios (approximate)
        aic_weight_ann = np.exp(-0.5 * delta_aic) / (1 + np.exp(-0.5 * abs(delta_aic)))
        aic_weight_theory = np.exp(-0.5 * (-delta_aic)) / (1 + np.exp(-0.5 * abs(delta_aic)))

        results = {
            # Likelihood-based metrics
            "log_likelihood_ann": log_likelihood_ann,
            "log_likelihood_theory": log_likelihood_theory,
            "sigma_ann": sigma_ann,
            "sigma_theory": sigma_theory,
            # Information criteria
            "aic_ann": aic_ann,
            "aic_theory": aic_theory,
            "bic_ann": bic_ann,
            "bic_theory": bic_theory,
            "waic_ann": waic_ann,
            "waic_theory": waic_theory,
            # Model comparison
            "delta_aic": delta_aic,
            "delta_bic": delta_bic,
            "delta_waic": delta_waic,
            "aic_weight_ann": aic_weight_ann,
            "aic_weight_theory": aic_weight_theory,
            # Basic metrics
            "rmse_ann": np.sqrt(np.mean((y_true - y_pred_ann) ** 2)),
            "rmse_theory": np.sqrt(np.mean((y_true - y_pred_theory) ** 2)),
            "mae_ann": np.mean(np.abs(y_true - y_pred_ann)),
            "mae_theory": np.mean(np.abs(y_true - y_pred_theory)),
            "sample_size": sample_size,
            "n_params_ann": n_params_ann,
            "n_params_theory": n_params_theory,
        }

        logger.info(
            f"Information criteria computed: AIC(ANN={aic_ann:.1f}, Theory={aic_theory:.1f})"
        )
        logger.info(
            f"Model preference: ΔAIC={delta_aic:.1f}, ΔBIC={delta_bic:.1f}, ΔWAIC={delta_waic:.1f}"
        )

        return results

    def statistical_tests(
        self,
        y_true: np.ndarray,
        y_pred_ann: np.ndarray,
        y_pred_theory: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Perform statistical hypothesis tests for model comparison.

        Args:
            y_true: True values
            y_pred_ann: Neural network predictions
            y_pred_theory: Theoretical model predictions
            uncertainties: Prediction uncertainties (for weighted tests)

        Returns:
            Dictionary with statistical test results
        """
        logger.info("Performing statistical comparison tests...")

        # Basic performance metrics
        errors_ann = y_true - y_pred_ann
        errors_theory = y_true - y_pred_theory

        # Diebold-Mariano test for predictive accuracy
        dm_stat, dm_pvalue = self._diebold_mariano_test(errors_ann, errors_theory)

        # Wilcoxon signed-rank test
        wilcoxon_stat, wilcoxon_pvalue = stats.wilcoxon(np.abs(errors_ann), np.abs(errors_theory))

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(y_pred_ann, y_pred_theory)

        # Kolmogorov-Smirnov test for distribution comparison
        ks_stat, ks_pvalue = stats.ks_2samp(y_pred_ann, y_pred_theory)

        # Cross-correlation analysis
        cross_corr = np.corrcoef(y_pred_ann, y_pred_theory)[0, 1]
        cross_corr_pvalue = self._correlation_significance(cross_corr, len(y_true))

        # If uncertainties provided, compute weighted statistics
        if uncertainties is not None:
            weighted_results = self._weighted_tests(
                y_true, y_pred_ann, y_pred_theory, uncertainties
            )
        else:
            weighted_results = {}

        results = {
            # Predictive performance
            "rmse_ann": np.sqrt(np.mean(errors_ann**2)),
            "rmse_theory": np.sqrt(np.mean(errors_theory**2)),
            "mae_ann": np.mean(np.abs(errors_ann)),
            "mae_theory": np.mean(np.abs(errors_theory)),
            # Statistical tests
            "diebold_mariano_stat": dm_stat,
            "diebold_mariano_pvalue": dm_pvalue,
            "wilcoxon_stat": wilcoxon_stat,
            "wilcoxon_pvalue": wilcoxon_pvalue,
            "t_statistic": t_stat,
            "t_pvalue": t_pvalue,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue,
            # Correlation analysis
            "cross_correlation": cross_corr,
            "cross_correlation_pvalue": cross_corr_pvalue,
            # Sample size
            "sample_size": len(y_true),
            # Significance summary
            "significant_dm": dm_pvalue < self.significance_level,
            "significant_wilcoxon": wilcoxon_pvalue < self.significance_level,
            "significant_ttest": t_pvalue < self.significance_level,
            "significant_kstest": ks_pvalue < self.significance_level,
            "significant_correlation": cross_corr_pvalue < self.significance_level,
        }

        results.update(weighted_results)

        # Summary
        logger.info(f"Statistical tests completed:")
        logger.info(f"  Diebold-Mariano: {dm_stat:.3f} (p={dm_pvalue:.3f})")
        logger.info(f"  Wilcoxon: {wilcoxon_stat:.3f} (p={wilcoxon_pvalue:.3f})")
        logger.info(f"  Cross-correlation: {cross_corr:.3f} (p={cross_corr_pvalue:.3f})")

        return results

    def _diebold_mariano_test(
        self, errors1: np.ndarray, errors2: np.ndarray
    ) -> Tuple[float, float]:
        """Diebold-Mariano test for equal predictive accuracy.

        H0: E[loss1^2] = E[loss2^2]
        HA: E[loss1^2] ≠ E[loss2^2]
        """

        loss_diff = errors1**2 - errors2**2
        mean_diff = np.mean(loss_diff)
        var_diff = np.var(loss_diff, ddof=1)

        # Test statistic (assuming asymptotic normality)
        dm_stat = mean_diff / np.sqrt(var_diff / len(loss_diff))
        p_value = 2 * (1 - stats.norm.cdf(abs(dm_stat)))

        return dm_stat, p_value

    def _correlation_significance(self, correlation: float, n_samples: int) -> float:
        """Test if correlation coefficient is statistically significant."""

        # Fisher's z-transformation
        z_corr = 0.5 * np.log((1 + correlation) / (1 - correlation))
        se_z = 1 / np.sqrt(n_samples - 3)

        # Two-sided test
        z_score = z_corr / se_z
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

        return p_value

    def _weighted_tests(
        self,
        y_true: np.ndarray,
        y_pred_ann: np.ndarray,
        y_pred_theory: np.ndarray,
        uncertainties: np.ndarray,
    ) -> Dict[str, float]:
        """Perform uncertainty-weighted statistical tests."""

        logger.info("Computing uncertainty-weighted tests...")

        # Weighted errors
        errors_ann = y_true - y_pred_ann
        errors_theory = y_true - y_pred_theory

        # Inverse variance weighting
        weights_ann = 1.0 / (uncertainties**2 + 1e-8)
        weights_theory = np.ones_like(errors_theory)  # Theory has no uncertainty

        # Normalize weights
        weights_ann = weights_ann / np.sum(weights_ann)
        weights_theory = weights_theory / np.sum(weights_theory)

        # Weighted means
        weighted_error_ann = np.sum(weights_ann * np.abs(errors_ann))
        weighted_error_theory = np.sum(weights_theory * np.abs(errors_theory))

        # Weighted Diebold-Mariano
        loss_diff = errors_ann**2 - errors_theory**2
        weighted_loss_diff = np.sum(weights_ann * loss_diff)
        weighted_dm_stat = weighted_loss_diff / np.sqrt(np.sum(weights_ann * loss_diff**2))

        return {
            "weighted_mae_ann": weighted_error_ann,
            "weighted_mae_theory": weighted_error_theory,
            "weighted_diebold_mariano": weighted_dm_stat,
            "mean_uncertainty": np.mean(uncertainties),
            "effective_sample_size": np.sum(weights_ann),
        }

    def consistency_analysis(
        self,
        y_true: np.ndarray,
        y_pred_ann: np.ndarray,
        y_pred_theory: np.ndarray,
        ann_uncertainties: Optional[np.ndarray] = None,
        theory_uncertainties: Optional[np.ndarray] = None,
        confidence_levels: list = [0.683, 0.955],  # 1σ and 2σ
    ) -> Dict[str, Any]:
        """Perform consistency analysis between ANN and theory.

        Args:
            y_true: True values
            y_pred_ann: Neural network predictions
            y_pred_theory: Theoretical model predictions
            ann_uncertainties: ANN prediction uncertainties
            theory_uncertainties: Theoretical model uncertainties
            confidence_levels: Confidence levels for analysis

        Returns:
            Dictionary with consistency metrics
        """
        logger.info("Performing consistency analysis...")

        results = {}

        for level in confidence_levels:
            from scipy import stats

            z_score = stats.norm.ppf((1 + level) / 2)

            # Tension metric
            tension = np.abs(y_pred_ann - y_pred_theory)

            if ann_uncertainties is not None and theory_uncertainties is not None:
                combined_uncertainty = np.sqrt(ann_uncertainties**2 + theory_uncertainties**2)
                tension_normalized = tension / combined_uncertainty
            elif ann_uncertainties is not None:
                tension_normalized = tension / ann_uncertainties
            else:
                tension_normalized = tension / theory_uncertainties

            # Consistency check
            is_consistent = (
                tension <= z_score * combined_uncertainty
                if "combined_uncertainty" in locals()
                else tension <= z_score * ann_uncertainties
                if ann_uncertainties is not None
                else tension <= z_score * theory_uncertainties
            )

            # Coverage analysis
            if ann_uncertainties is not None:
                theory_in_interval = (
                    np.abs(y_pred_theory - y_pred_ann) <= z_score * ann_uncertainties
                )
                coverage_theory_in_ann = np.mean(theory_in_interval)

            results[f"level_{level:.3f}"] = {
                "tension": np.mean(tension),
                "tension_normalized": np.mean(tension_normalized),
                "z_score": z_score,
                "is_consistent": np.mean(is_consistent),
                "coverage_theory_in_ann": coverage_theory_in_ann
                if ann_uncertainties is not None
                else None,
                "tension_fraction": np.mean(
                    tension
                    > z_score
                    * (
                        ann_uncertainties
                        if ann_uncertainties is not None
                        else theory_uncertainties
                    )
                ),
            }

        logger.info(
            f"Consistency analysis completed for {len(confidence_levels)} confidence levels"
        )

        return results

    def predictive_performance_comparison(
        self,
        y_true: np.ndarray,
        y_pred_ann: np.ndarray,
        y_pred_theory: np.ndarray,
        ann_uncertainties: Optional[np.ndarray] = None,
        cv_folds: int = 5,
    ) -> Dict[str, Any]:
        """Cross-validated predictive performance comparison.

        Args:
            y_true: True values
            y_pred_ann: Neural network predictions
            y_pred_theory: Theoretical model predictions
            ann_uncertainties: ANN prediction uncertainties
            cv_folds: Number of CV folds

        Returns:
            Dictionary with CV comparison results
        """
        logger.info(f"Performing {cv_folds}-fold CV comparison...")

        # Simple time-series split (maintain temporal order)
        n_samples = len(y_true)
        fold_size = n_samples // cv_folds

        cv_results = {
            "ann_rmse_folds": [],
            "theory_rmse_folds": [],
            "ann_mae_folds": [],
            "theory_mae_folds": [],
            "coverage_folds": [],
        }

        for fold in range(cv_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < cv_folds - 1 else n_samples

            # Test set
            y_test = y_true[start_idx:end_idx]
            y_pred_ann_test = y_pred_ann[start_idx:end_idx]
            y_pred_theory_test = y_pred_theory[start_idx:end_idx]

            if ann_uncertainties is not None:
                unc_test = ann_uncertainties[start_idx:end_idx]
                coverage = np.mean(
                    np.abs(y_pred_theory_test - y_pred_ann_test)
                    <= stats.norm.ppf(0.8413) * unc_test
                )
            else:
                coverage = None

            # Training set (remaining data)
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[start_idx:end_idx] = False

            y_train = y_true[train_mask]
            y_pred_ann_train = y_pred_ann[train_mask]
            y_pred_theory_train = y_pred_theory[train_mask]

            # Compute metrics
            rmse_ann = np.sqrt(np.mean((y_test - y_pred_ann_test) ** 2))
            rmse_theory = np.sqrt(np.mean((y_test - y_pred_theory_test) ** 2))
            mae_ann = np.mean(np.abs(y_test - y_pred_ann_test))
            mae_theory = np.mean(np.abs(y_test - y_pred_theory_test))

            cv_results["ann_rmse_folds"].append(rmse_ann)
            cv_results["theory_rmse_folds"].append(rmse_theory)
            cv_results["ann_mae_folds"].append(mae_ann)
            cv_results["theory_mae_folds"].append(mae_theory)
            cv_results["coverage_folds"].append(coverage)

        # Summary statistics
        results = {
            "cv_folds": cv_folds,
            "ann_rmse_mean": np.mean(cv_results["ann_rmse_folds"]),
            "ann_rmse_std": np.std(cv_results["ann_rmse_folds"]),
            "theory_rmse_mean": np.mean(cv_results["theory_rmse_folds"]),
            "theory_rmse_std": np.std(cv_results["theory_rmse_folds"]),
            "ann_mae_mean": np.mean(cv_results["ann_mae_folds"]),
            "ann_mae_std": np.std(cv_results["ann_mae_folds"]),
            "theory_mae_mean": np.mean(cv_results["theory_mae_folds"]),
            "theory_mae_std": np.std(cv_results["theory_mae_folds"]),
            "coverage_mean": np.mean([c for c in cv_results["coverage_folds"] if c is not None]),
            "rmse_improvement": (
                np.mean(cv_results["ann_rmse_folds"]) - np.mean(cv_results["theory_rmse_folds"])
            )
            / np.mean(cv_results["theory_rmse_folds"]),
            "significance_test": self._statistical_tests(
                y_true, y_pred_ann, y_pred_theory, ann_uncertainties
            ),
        }

        logger.info(
            f"CV comparison completed: ANN RMSE={results['ann_rmse_mean']:.3f}±{results['ann_rmse_std']:.3f}, "
            f"Theory RMSE={results['theory_rmse_mean']:.3f}±{results['theory_rmse_std']:.3f}"
        )

        return results
