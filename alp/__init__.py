"""
Astro Layer Perceptron
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
A modular framework for stellar spectroscopy analysis and 
Artificial Neural Networks inference using shell representations.

Subpackages:
    - physics:   Physical modeling tools (CCF, shells, periodograms)
    - data:      Data loading, preprocessing, and scaling
    - networks:  Deep learning architectures (CNNs, VAEs, etc.)
    - utils:     Logging configuration and global utilities
"""

__author__ = "Isidro Gomez-Vargas"
__version__ = "1.0.0"

# --- Public Submodules ---
from . import physics
from . import data
from . import networks
from . import utils

# --- Core Logger ---
from .utils import logger

__all__ = [
    "physics",
    "data",
    "networks",
    "utils",
    "logger",
]
