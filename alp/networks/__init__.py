"""
ALP
----------------------------------------------------------
2025
by Isidro Gomez-Vargas
----------------------------------------------------------
"""

# --- Base Classes ---
from .base_networks import SupervisedNET

# --- Custom Layers and Tensor Utilities ---
from .net_blocks import MCDropout

# --- Model Architectures ---
from .mlp import MLP
from .losses import heteroscedastic_gaussian_nll
from .uncertainty import UncertaintyQuantifier

# --- Hyperparameter Optimization ---
from .optuna_optimizer import OptunaOptimizer, quick_optimize
from .trainer import HyperparameterOptimizer, ModelTrainer

# --- Uncertainty Analysis ---
from .uncertainty_analyzer import UncertaintyAnalyzer, load_and_analyze_model
from .conformal_prediction import (
    ConformalizedQuantileRegression,
    QuantileRegressionNN,
    create_cqr_for_sigma,
)

__all__ = [
    # Base Classes
    "SupervisedNET",
    # Custom Layers and Utilities
    "MCDropout",
    # Model Architectures
    "MLP",
    # Loss Functions and Uncertainty
    "heteroscedastic_gaussian_nll",
    "UncertaintyQuantifier",
    # Hyperparameter Optimization
    "OptunaOptimizer",
    "quick_optimize",
    "HyperparameterOptimizer",
    "ModelTrainer",
    # Uncertainty Analysis
    "UncertaintyAnalyzer",
    "load_and_analyze_model",
    "ConformalizedQuantileRegression",
    "QuantileRegressionNN",
    "create_cqr_for_sigma",
]
