"""
DopplerIANN Utility Module
----------------------------------------------------------
2025
by Isidro Gomez-Vargas
----------------------------------------------------------
Contains general-purpose utility tools such as logging configuration.
"""

from .logger_config import logger, setup_logging

# --- GPU Configuration ---
from .gpu_config import (
    configure_gpu,
    set_random_seed_safely,
    setup_tensorflow_for_training,
    handle_cuda_error,
)

# --- Visualization ---
from .visualization import (
    plot_lsst_results,
    plot_training_history,
    plot_uncertainty_analysis,
    create_lcdm_model,
)

__all__ = [
    "logger",
    "setup_logging",
    "configure_gpu",
    "set_random_seed_safely",
    "setup_tensorflow_for_training",
    "handle_cuda_error",
    "plot_lsst_results",
    "plot_training_history",
    "plot_uncertainty_analysis",
    "create_lcdm_model",
]
