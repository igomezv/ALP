"""
Astro Layer Perceptron
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
A deep learning framework tailored for astrophysical data modeling.

Subpackages:
    - physics:   Physical modeling tools
    - data:      Data loading, preprocessing and augmentation
    - networks:  Deep learning architectures 
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
