import os
import subprocess
from pathlib import Path
import logging
from typing import Union, List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.python.distribute.distribute_lib import Strategy

# === Constants and Configuration ===

# Path to nvidia-smi (adjust if needed)
NVIDIA_SMI_PATH = Path("/usr/bin/nvidia-smi")

# ANSI color codes for terminal output.
ANSI_RED = "\033[31m"
ANSI_YELLOW = "\033[33m"
ANSI_GREEN = "\033[32m"
ANSI_RESET = "\033[0m"

# Default thresholds.
DEFAULT_MIN_MEMORY_MB = 5000
DEFAULT_MAX_UTILIZATION_PERCENTAGE = 80

# Configure logging once.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# === GPU Query and Status Functions ===

def get_gpu_memory_info() -> Optional[List[dict]]:
    """
    Queries nvidia-smi to obtain GPU information.

    Returns:
        A list of dictionaries with keys: 'index', 'total', 'used', 'free', 'utilization'.
        Returns None if nvidia-smi is not available or fails.
    """
    if not NVIDIA_SMI_PATH.exists():
        logging.error("nvidia-smi not found at %s", NVIDIA_SMI_PATH)
        return None

    try:
        output = subprocess.check_output(
            [str(NVIDIA_SMI_PATH),
             "--query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
    except subprocess.CalledProcessError as e:
        logging.error("Error running nvidia-smi: %s", e.output)
        return None

    gpu_info_list = []
    for line in output.strip().split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        gpu_info_list.append({
            "index": parts[0],
            "total": int(parts[1]),
            "used": int(parts[2]),
            "free": int(parts[3]),
            "utilization": int(parts[4])
        })
    return gpu_info_list


def color_for_percentage(percentage: float) -> str:
    """
    Returns an ANSI color code based on memory usage percentage.
    
    Args:
        percentage (float): Memory usage percentage.

    Returns:
        str: ANSI color code.
    """
    if percentage >= 80:
        return ANSI_RED
    elif percentage >= 50:
        return ANSI_YELLOW
    else:
        return ANSI_GREEN


def print_gpu_status(min_required_memory: Optional[int] = None) -> None:
    """
    Prints a slimmed-down, color-coded table of GPU status.

    Args:
        min_required_memory (Optional[int]): If provided, GPUs with free memory lower than
            this value will have their free memory printed in red.
    """
    gpu_info_list = get_gpu_memory_info()
    if gpu_info_list is None:
        print("nvidia-smi not available; cannot display GPU status.")
        return

    header = f"{'GPU':<5} {'Total (MB)':<12} {'Used (MB)':<12} {'Free (MB)':<12} {'Util (%)':<10}"
    print(header)
    print("-" * len(header))

    for gpu_info in gpu_info_list:
        usage_pct = (gpu_info["used"] / gpu_info["total"]) * 100 if gpu_info["total"] > 0 else 0
        usage_color = color_for_percentage(usage_pct)
        if min_required_memory is not None:
            free_color = ANSI_RED if gpu_info["free"] < min_required_memory else ANSI_GREEN
        else:
            free_color = ANSI_GREEN  # default to green if no threshold is given

        row = (
            f"{gpu_info['index']:<5} "
            f"{gpu_info['total']:<12} "
            f"{gpu_info['used']:<12} "
            f"{free_color}{gpu_info['free']:<12}{ANSI_RESET} "
            f"{usage_color}{usage_pct:>8.1f}%{ANSI_RESET}"
        )
        print(row)


# === TensorFlow Environment Setup ===

def setup_cuda(device_num: str, max_memory_limit: int, logging_level: int = logging.WARNING) -> tf.distribute.Strategy:
    """
    Configures CUDA for TensorFlow by restricting visible GPUs to only the ones specified,
    setting a memory limit per GPU, and configuring thread options.
    
    Args:
        device_num (str): A comma-separated list of GPU indices to be used (e.g. "0" or "0,2").
        max_memory_limit (int): Maximum memory (in MB) per GPU.
        logging_level (int): Logging level for TensorFlow logs.
        
    Returns:
        tf.distribute.MirroredStrategy: The distribution strategy for multi-GPU training.
    """
    # Log to file.
    logging.basicConfig(filename='tensorflow_setup.log', level=logging_level)
    
    # Get all physical GPUs.
    all_gpus = tf.config.list_physical_devices('GPU')
    if not all_gpus:
        logging.warning("No physical GPUs detected; running on CPU.")
    else:
        try:
            # Parse allowed GPU indices from the comma-separated string.
            allowed_indices = [int(idx.strip()) for idx in device_num.split(",")]
        except ValueError:
            raise ValueError("device_num should be a comma-separated string of integers")
        
        # Restrict visible GPUs to only those specified.
        visible_gpus = [all_gpus[i] for i in allowed_indices if i < len(all_gpus)]
        tf.config.set_visible_devices(visible_gpus, 'GPU')
        
        # Configure each visible GPU with the desired memory limit.
        for gpu in visible_gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=max_memory_limit)]
            )
    
    # Set threading options and reduce verbosity.
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    
    # Create a MirroredStrategy. Now only the restricted GPUs are visible.
    strategy = tf.distribute.MirroredStrategy()
    logging.info(f"Visible GPUs after restriction: {tf.config.list_physical_devices('GPU')}")
    return strategy


def get_tf_memory_usage() -> int:
    """
    Returns the current GPU memory usage (in MB) for the first visible GPU.
    
    Returns:
        int: Current GPU memory usage.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return 0
    device_index = int(gpus[0].name.split(":")[-1])
    device_name = f"GPU:{device_index}"
    memory_info = tf.config.experimental.get_memory_info(device_name)
    return memory_info["current"] // (1024 * 1024)


def find_available_GPUs(
    min_memory_MB: Optional[int] = None,
    max_utilization_percentage: float = DEFAULT_MAX_UTILIZATION_PERCENTAGE,
    max_needed: int = -1
) -> str:
    """
    Selects available GPUs based on free memory and utilization criteria.
    
    Args:
        min_memory_MB (Optional[int]): Minimum free memory required (in MB).
        max_utilization_percentage (float): Maximum allowed GPU utilization (%).
        max_needed (int): Maximum number of GPUs to return (-1 for no limit).
        
    Returns:
        str: A comma-separated string of GPU indices.
        
    Raises:
        RuntimeError: If no GPUs are found or if none meet the criteria.
    """
    physical_gpus = tf.config.list_physical_devices('GPU')
    if not physical_gpus:
        raise RuntimeError("No GPUs found on the system.")

    gpu_info_list = get_gpu_memory_info()
    if gpu_info_list is None:
        raise RuntimeError("Failed to retrieve GPU info; ensure nvidia-smi is installed.")

    available = []
    for gpu_info in gpu_info_list:
        if (min_memory_MB is None or gpu_info["free"] > min_memory_MB) and \
           (gpu_info["utilization"] < max_utilization_percentage):
            available.append(gpu_info["index"])

    if not available:
        print("\nAll GPUs are currently busy or do not meet the memory requirements:")
        print_gpu_status(min_required_memory=min_memory_MB)
        raise RuntimeError("All GPUs are busy or insufficient memory is available.")

    if max_needed != -1:
        available = available[:max_needed]

    logging.info("Available GPUs: %s", available)
    return ",".join(available)


# Module-level variable to cache the strategy.
_global_strategy = None

def env(
    min_gpu_memory_mb: int = 5000,
    max_gpu_utilization_percentage: float = 80,
    num_gpus_to_request: int = 1,
    memory_to_allocate_tf: int = 3000,
    gpus: Union[str, int, List[Union[int, str]], None] = None
) -> tf.distribute.Strategy:
    """
    Sets up the TensorFlow environment with the requested GPU(s).

    If a distribution strategy has already been created, its scope is returned.
    Otherwise, the function automatically selects GPUs based on the provided
    criteria and creates a new MirroredStrategy.

    Args:
        min_gpu_memory_mb (int): Minimum free memory per GPU (MB).
        max_gpu_utilization_percentage (float): Maximum allowed GPU utilization (%).
        num_gpus_to_request (int): Number of GPUs requested.
        memory_to_allocate_tf (int): Memory limit per GPU for TensorFlow (MB).
        gpus (Union[int, str, List[int]]): Optional specification of GPU(s) to use.

    Returns:
        tf.distribute.Strategy: A TensorFlow distribution strategy scope.
    """
    global _global_strategy

    if _global_strategy is not None:
        logging.info("Using previously created distribution strategy.")
        return _global_strategy.scope()

    # Process the 'gpus' parameter.
    if gpus is not None:
        if isinstance(gpus, int):
            gpus = str(gpus)
        elif isinstance(gpus, (list, tuple)):
            gpus = ",".join([str(gpu) for gpu in gpus])
        elif not isinstance(gpus, str):
            raise ValueError("gpus should be int, str, or list/tuple of int or str")
    else:
        gpus = find_available_GPUs(
            min_memory_MB=min_gpu_memory_mb, 
            max_utilization_percentage=max_gpu_utilization_percentage,
            max_needed=num_gpus_to_request
        )

    available_gpu_list = gpus.split(",") if gpus else []
    if not available_gpu_list:
        raise RuntimeError("No GPUs available (they may be busy or not present).")
    elif len(available_gpu_list) < num_gpus_to_request:
        logging.warning(
            f"Requested {num_gpus_to_request} GPUs, but only {len(available_gpu_list)} are available."
        )

    # Create the strategy with the chosen GPUs.
    _global_strategy = setup_cuda(
        device_num=gpus, 
        max_memory_limit=memory_to_allocate_tf, 
        logging_level=logging.WARNING
    )
    
    return _global_strategy.scope()