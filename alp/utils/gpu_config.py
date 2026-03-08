#!/usr/bin/env python

"""
GPU Configuration Utility for ALP Networks
----------------------------------------------------------
2025
by Isidro Gomez-Vargas (isidro.gomezvargas@unige.ch)
----------------------------------------------------------
Provides GPU setup and CUDA error handling for TensorFlow operations.
"""

import os
import numpy as np
import tensorflow as tf
from typing import Optional
from .logger_config import logger


def configure_gpu(
    memory_growth: bool = True,
    allow_memory_growth: bool = True,
    set_visible_devices: Optional[str] = None,
) -> bool:
    """
    Configure GPU settings to avoid CUDA errors.

    Parameters
    ----------
    memory_growth : bool, optional
        Enable memory growth to prevent GPU memory allocation issues
    allow_memory_growth : bool, optional
        Allow GPU memory to grow as needed
    set_visible_devices : str, optional
        Comma-separated list of GPU devices to use (e.g., "0", "0,1")

    Returns
    -------
    bool
        True if GPU configuration was successful, False otherwise
    """
    try:
        # Set environment variables for better CUDA compatibility
        os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if allow_memory_growth else "false"
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

        # Configure GPU devices
        gpus = tf.config.list_physical_devices("GPU")

        if gpus:
            logger.info(f"Found {len(gpus)} GPU device(s)")

            # Set specific GPUs if requested
            if set_visible_devices is not None:
                tf.config.set_visible_devices(
                    [gpus[int(i)] for i in set_visible_devices.split(",") if int(i) < len(gpus)],
                    "GPU",
                )
                gpus = tf.config.list_physical_devices("GPU")

            # Configure memory growth
            if memory_growth or allow_memory_growth:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        logger.info(f"Enabled memory growth for {gpu.name}")
                    except RuntimeError as e:
                        logger.warning(f"Failed to set memory growth for {gpu.name}: {e}")

            logger.info("GPU configuration completed successfully")
            return True
        else:
            logger.info("No GPU devices found, using CPU")
            return False

    except Exception as e:
        logger.error(f"GPU configuration failed: {e}")
        return False


def set_random_seed_safely(seed: int = 42) -> bool:
    """
    Set random seeds in a GPU-safe manner to avoid CUDA errors.

    Parameters
    ----------
    seed : int, optional
        Random seed value (default 42)

    Returns
    -------
    bool
        True if seed was set successfully, False otherwise
    """
    try:
        # Set random seeds in the correct order
        import random

        random.seed(seed)
        np.random.seed(seed)

        # Try TensorFlow seed setting with fallback
        try:
            tf.keras.utils.set_random_seed(seed)
            logger.info(f"Set TensorFlow random seed to {seed}")
        except Exception as tf_error:
            logger.warning(f"tf.keras.utils.set_random_seed failed: {tf_error}")
            # Fallback method
            tf.random.set_seed(seed)
            logger.info(f"Set TensorFlow random seed using fallback method: {seed}")

        # Ensure deterministic operations where possible
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["PYTHONHASHSEED"] = str(seed)

        return True

    except Exception as e:
        logger.error(f"Failed to set random seed safely: {e}")
        return False


def setup_tensorflow_for_training(
    seed: int = 42, configure_gpu_settings: bool = True, force_cpu: bool = False
) -> bool:
    """
    Complete TensorFlow setup for training with error handling.

    Parameters
    ----------
    seed : int, optional
        Random seed for reproducibility (default 42)
    configure_gpu_settings : bool, optional
        Whether to configure GPU settings (default True)
    force_cpu : bool, optional
        Force CPU usage to avoid CUDA issues (default False)

    Returns
    -------
    bool
        True if setup was successful, False otherwise
    """
    success = True

    logger.info("Setting up TensorFlow for training...")

    # Force CPU if requested or if GPU issues are detected
    if force_cpu:
        logger.info("Forcing CPU usage to avoid CUDA issues")
        tf.config.set_visible_devices([], "GPU")
        configure_gpu_settings = False

    # Configure GPU first
    gpu_config_success = True
    if configure_gpu_settings:
        gpu_success = configure_gpu()
        if not gpu_success:
            logger.warning("GPU configuration failed, falling back to CPU")
            tf.config.set_visible_devices([], "GPU")
            gpu_config_success = False

    # Set random seeds safely (with CPU fallback)
    try:
        seed_success = set_random_seed_safely(seed)
        if not seed_success:
            logger.warning("Random seed setting failed, training may not be reproducible")
            success = False
    except Exception as e:
        logger.warning(f"Random seed setting failed: {e}, trying CPU-only approach")
        # CPU-only seed setting
        try:
            import random

            random.seed(seed)
            np.random.seed(seed)
            tf.random.set_seed(seed)
            logger.info("CPU-only seed setting successful")
        except Exception as fallback_error:
            logger.error(f"Even CPU-only seed setting failed: {fallback_error}")
            success = False

    # Log TensorFlow info
    logger.info(f"TensorFlow Version: {tf.__version__}")
    logger.info(f"Available devices: {len(tf.config.list_physical_devices())}")

    gpu_devices = tf.config.list_physical_devices("GPU")
    cpu_devices = tf.config.list_physical_devices("CPU")

    visible_devices = tf.config.get_visible_devices()
    logger.info(f"Visible devices: {[d.name for d in visible_devices]}")

    if gpu_devices and not force_cpu and gpu_config_success:
        logger.info(f"GPU devices available: {[gpu.name for gpu in gpu_devices]}")
        logger.warning("If you encounter CUDA errors, run with: force_cpu=True")
    else:
        logger.info("Using CPU only")

    if success:
        logger.info("TensorFlow setup completed successfully")
    else:
        logger.warning("TensorFlow setup completed with warnings")

    return success


def handle_cuda_error(error: Exception) -> str:
    """
    Provide guidance for common CUDA errors.

    Parameters
    ----------
    error : Exception
        The CUDA error that occurred

    Returns
    -------
    str
        Suggested solution for the error
    """
    error_str = str(error).lower()

    if "cuda_error_invalid_handle" in error_str:
        return (
            "CUDA_ERROR_INVALID_HANDLE: Try restarting the Python process. "
            "This often occurs when GPU context is corrupted. "
            "Alternatively, run with: TF_FORCE_GPU_ALLOW_GROWTH=true python script.py"
        )
    elif "out of memory" in error_str:
        return (
            "GPU out of memory: Try reducing batch size or enabling memory growth. "
            "Add this to your script: configure_gpu(memory_growth=True)"
        )
    elif "cuLaunchKernel" in error_str:
        return (
            "cuLaunchKernel error: This is often a seed-related CUDA issue. "
            "Use setup_tensorflow_for_training() instead of direct seed setting."
        )
    elif "floorMod" in error_str:
        return (
            "FloorMod CUDA error: This is related to TensorFlow random seed casting. "
            "Use set_random_seed_safely() instead of tf.keras.utils.set_random_seed()"
        )
    else:
        return f"Unknown CUDA error: {error}. Try restarting the process or running on CPU only."
