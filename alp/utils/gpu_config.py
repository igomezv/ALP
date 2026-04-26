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
from typing import Optional, Tuple
from contextlib import contextmanager
from .logger_config import logger


# Set critical environment variables BEFORE importing TensorFlow modules
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_device_name() -> str:
    """
    Get the current device being used (GPU or CPU).

    Returns
    -------
    str
        Device name (e.g., "GPU:0", "CPU:0")
    """
    try:
        visible_devices = tf.config.get_visible_devices()
        if visible_devices:
            return visible_devices[0].name
        return "CPU:0"
    except Exception as e:
        logger.warning(f"Failed to get device name: {e}, defaulting to CPU")
        return "CPU:0"


def is_gpu_available() -> bool:
    """
    Check if GPU is available and configured.

    Returns
    -------
    bool
        True if GPU is available and configured, False otherwise
    """
    try:
        gpus = tf.config.list_physical_devices("GPU")
        visible_gpus = tf.config.get_visible_devices("GPU")
        return len(gpus) > 0 and len(visible_gpus) > 0
    except Exception as e:
        logger.debug(f"Error checking GPU availability: {e}")
        return False


@contextmanager
def device_context(device: Optional[str] = None):
    """
    Context manager for explicit device placement with fallback.

    Parameters
    ----------
    device : str, optional
        Device to use (e.g., "GPU:0", "CPU:0"). If None, uses default device.

    Yields
    ------
    str
        The device being used

    Example
    -------
    >>> with device_context("GPU:0") as device:
    ...     model = create_model()  # Will run on GPU or CPU if unavailable
    """
    if device is None:
        device = get_device_name()

    try:
        with tf.device(device):
            yield device
    except RuntimeError as e:
        if "GPU" in device and ("not found" in str(e) or "CUDA" in str(e)):
            logger.warning(f"Device {device} not available, falling back to CPU: {e}")
            with tf.device("CPU:0"):
                yield "CPU:0"
        else:
            raise


def create_safe_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = False,
    buffer_size: Optional[int] = None,
    disable_parallelization: bool = True,
) -> tf.data.Dataset:
    """
    Create a safe tf.data.Dataset with automatic GPU-to-CPU fallback.

    This function ensures datasets are created safely with parallel processing disabled
    on GPU to avoid memory fragmentation and CUDA errors. Falls back to CPU data loading
    if GPU issues occur.

    Parameters
    ----------
    x : np.ndarray
        Input features
    y : np.ndarray
        Target values
    batch_size : int, optional
        Batch size (default 32)
    shuffle : bool, optional
        Whether to shuffle the dataset (default False)
    buffer_size : int, optional
        Shuffle buffer size (default None, uses dataset size)
    disable_parallelization : bool, optional
        Disable parallel processing for data loading (default True, recommended for GPU)

    Returns
    -------
    tf.data.Dataset
        Safe dataset with disabled parallelization on GPU

    Example
    -------
    >>> x_train = np.random.randn(1000, 10)
    >>> y_train = np.random.randn(1000, 1)
    >>> dataset = create_safe_dataset(x_train, y_train, batch_size=32)
    """
    try:
        # Use CPU for data loading to avoid GPU memory fragmentation
        with tf.device("CPU:0"):
            # Create dataset from numpy arrays
            dataset = tf.data.Dataset.from_tensor_slices((x, y))

            # Shuffle if requested
            if shuffle:
                buffer_sz = buffer_size if buffer_size is not None else len(x)
                dataset = dataset.shuffle(buffer_size=buffer_sz)

            # Batch the dataset
            dataset = dataset.batch(batch_size)

            # Disable parallelization if requested (GPU-safe)
            if disable_parallelization or is_gpu_available():
                num_parallel_calls = 1
                logger.debug(
                    f"Creating dataset with num_parallel_calls=1 (GPU-safe mode). "
                    f"Batch size: {batch_size}, Shuffle: {shuffle}"
                )
            else:
                num_parallel_calls = tf.data.AUTOTUNE
                logger.debug(
                    f"Creating dataset with num_parallel_calls=AUTOTUNE. "
                    f"Batch size: {batch_size}, Shuffle: {shuffle}"
                )

            # Prefetch data for performance
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

            return dataset

    except Exception as e:
        logger.error(f"Failed to create safe dataset: {e}")
        logger.warning("Falling back to numpy arrays for training")
        # Return as-is; training loop should handle both Dataset and array formats
        raise
=======
# Set critical environment variables BEFORE importing TensorFlow modules
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
>>>>>>> Stashed changes


def configure_gpu(
    memory_growth: bool = True,
    allow_memory_growth: bool = True,
    set_visible_devices: Optional[str] = None,
    retry_count: int = 3,
) -> bool:
    """
    Configure GPU settings to avoid CUDA errors with automatic fallback to CPU.

    Parameters
    ----------
    memory_growth : bool, optional
        Enable memory growth to prevent GPU memory allocation issues
    allow_memory_growth : bool, optional
        Allow GPU memory to grow as needed
    set_visible_devices : str, optional
        Comma-separated list of GPU devices to use (e.g., "0", "0,1")
    retry_count : int, optional
        Number of retry attempts for GPU configuration (default 3)

    Returns
    -------
    bool
        True if GPU configuration was successful, False otherwise (falls back to CPU)
    """
    for attempt in range(retry_count):
        try:
            # Set environment variables for better CUDA compatibility
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" if allow_memory_growth else "false"
            os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

            # Configure GPU devices
            gpus = tf.config.list_physical_devices("GPU")

            if gpus:
                logger.info(f"Found {len(gpus)} GPU device(s) (attempt {attempt + 1}/{retry_count})")

                # Set specific GPUs if requested
                if set_visible_devices is not None:
                    try:
                        device_indices = [int(i) for i in set_visible_devices.split(",")]
                        valid_indices = [i for i in device_indices if i < len(gpus)]
                        tf.config.set_visible_devices(
                            [gpus[i] for i in valid_indices],
                            "GPU",
                        )
                        gpus = tf.config.list_physical_devices("GPU")
                        logger.info(f"Configured {len(gpus)} GPU device(s): {[g.name for g in gpus]}")
                    except (ValueError, IndexError) as e:
                        logger.warning(f"Invalid GPU device specification '{set_visible_devices}': {e}")

                # Configure memory growth
                if memory_growth or allow_memory_growth:
                    for gpu in gpus:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                            logger.info(f"Enabled memory growth for {gpu.name}")
                        except RuntimeError as e:
                            logger.warning(f"Failed to set memory growth for {gpu.name}: {e}")
                            if attempt < retry_count - 1:
                                logger.info("Retrying GPU configuration...")
                                continue

                logger.info("GPU configuration completed successfully")
                return True
            else:
                logger.info("No GPU devices found, using CPU")
                return False

        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                logger.warning(f"GPU configuration failed (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    logger.info("Retrying GPU configuration...")
                    continue
                else:
                    logger.error("GPU configuration failed after all retries, falling back to CPU")
                    tf.config.set_visible_devices([], "GPU")
                    return False
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error during GPU configuration: {e}")
            return False

    # Fallback to CPU if all retries failed
    logger.warning("All GPU configuration attempts failed, falling back to CPU")
    tf.config.set_visible_devices([], "GPU")
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


def disable_dataset_parallelization() -> bool:
    """
    Disable TensorFlow dataset parallelization to avoid threadpool issues.

    This fixes the "use_unbounded_threadpool" attribute error that causes
    training to hang indefinitely.

    Returns
    -------
    bool
        True if configuration was successful
    """
    try:
        # Disable inter-op and intra-op parallelism
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

        logger.info("Disabled dataset parallelization to prevent threadpool issues")
        logger.info("Set thread limits: inter_op=1, intra_op=1")
        return True
    except Exception as e:
        logger.warning(f"Failed to disable dataset parallelization: {e}")
        return False


def setup_tensorflow_for_training(
    seed: int = 42, configure_gpu_settings: bool = True, force_cpu: bool = False
) -> Tuple[bool, str]:
    """
    Complete TensorFlow setup for training with automatic GPU-to-CPU fallback.

    This function ensures robust training setup by automatically falling back to CPU
    if GPU initialization fails. It handles seed setting safely and provides device info.

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
    tuple
        (success: bool, device: str)
        - success: True if setup was successful, False otherwise
        - device: The device being used (e.g., "GPU:0" or "CPU:0")
    """
    success = True
    device = "CPU:0"

    logger.info("Setting up TensorFlow for training...")

    # Disable dataset parallelization early to prevent hang issues
    disable_dataset_parallelization()

    # Force CPU if requested or if GPU issues are detected
    if force_cpu:
        logger.info("Forcing CPU usage as requested")
        tf.config.set_visible_devices([], "GPU")
        configure_gpu_settings = False
        device = "CPU:0"
    else:
        # Configure GPU first with automatic fallback
        if configure_gpu_settings:
            gpu_success = configure_gpu()
            if gpu_success:
                device = get_device_name()
                logger.info(f"Using device: {device}")
            else:
                logger.warning("GPU configuration failed, falling back to CPU")
                tf.config.set_visible_devices([], "GPU")
                device = "CPU:0"

    # Set random seeds safely (with CPU fallback)
    try:
        seed_success = set_random_seed_safely(seed)
        if not seed_success:
            logger.warning("Random seed setting had issues, training may not be fully reproducible")
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

    if gpu_devices and not force_cpu and is_gpu_available():
        logger.info(f"GPU devices available: {[gpu.name for gpu in gpu_devices]}")
        logger.info(f"Training will use: {device}")
        logger.warning("If you encounter CUDA errors during training, set force_cpu=True")
    else:
        logger.info("Training will use CPU only")

    if success:
        logger.info("TensorFlow setup completed successfully")
    else:
        logger.warning("TensorFlow setup completed with warnings")

    return success, device


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


def safe_model_fit(
    model: tf.keras.Model,
    x_train,
    y_train,
    x_val=None,
    y_val=None,
    **fit_kwargs
) -> Tuple[Optional[tf.keras.callbacks.History], bool]:
    """
    Safely train a model with automatic GPU-to-CPU fallback on CUDA errors.

    This wrapper attempts to train on GPU and automatically falls back to CPU
    if CUDA errors occur.

    Parameters
    ----------
    model : tf.keras.Model
        The model to train
    x_train : np.ndarray or tf.data.Dataset
        Training input data
    y_train : np.ndarray or tf.data.Dataset
        Training target data
    x_val : np.ndarray or tf.data.Dataset, optional
        Validation input data
    y_val : np.ndarray or tf.data.Dataset, optional
        Validation target data
    **fit_kwargs
        Additional keyword arguments to pass to model.fit()

    Returns
    -------
    tuple
        (history, success)
        - history: Training history (or None if training failed)
        - success: True if training succeeded, False otherwise

    Example
    -------
    >>> model = tf.keras.Sequential([...])
    >>> history, success = safe_model_fit(
    ...     model, x_train, y_train,
    ...     x_val=x_val, y_val=y_val,
    ...     epochs=100, batch_size=32
    ... )
    """
    try:
        # First attempt: Try with current device (GPU if available)
        logger.info(f"Training on {get_device_name()}...")
        
        if x_val is not None and y_val is not None:
            validation_data = (x_val, y_val)
        else:
            validation_data = None

        history = model.fit(
            x_train, y_train,
            validation_data=validation_data,
            **fit_kwargs
        )
        logger.info("Training completed successfully")
        return history, True

    except RuntimeError as e:
        if "CUDA" in str(e) or "GPU" in str(e) or "out of memory" in str(e):
            logger.warning(f"GPU error detected: {e}")
            logger.info(handle_cuda_error(e))

            # Check if we're already on CPU
            if get_device_name() == "CPU:0":
                logger.error("Already on CPU, cannot fallback further. Training failed.")
                return None, False

            # Fallback to CPU
            logger.warning("Falling back to CPU training...")
            tf.config.set_visible_devices([], "GPU")

            # Retry on CPU
            try:
                logger.info("Retrying training on CPU...")
                
                if x_val is not None and y_val is not None:
                    validation_data = (x_val, y_val)
                else:
                    validation_data = None

                history = model.fit(
                    x_train, y_train,
                    validation_data=validation_data,
                    **fit_kwargs
                )
                logger.info("Training completed successfully on CPU")
                return history, True

            except Exception as cpu_error:
                logger.error(f"Training failed even on CPU: {cpu_error}")
                return None, False
        else:
            # Non-GPU error, re-raise
            logger.error(f"Non-GPU training error: {e}")
            raise
    except Exception as e:
        logger.error(f"Unexpected training error: {e}")
        return None, False
