"""
Hardware configuration utilities for TensorFlow setup.

This module provides functions to configure TensorFlow for optimal 
performance on different hardware platforms (CPU, GPU, Apple Silicon).
"""

import platform
import tensorflow as tf
import logging
import os

logger = logging.getLogger(__name__)


def configure_hardware(config):
    """
    Configure hardware settings for optimal training performance.

    Args:
        config: Configuration dictionary

    Returns:
        dict: Hardware information and configuration
    """
    import platform
    import os
    import tensorflow as tf

    hardware_info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "architecture": platform.machine(),
        "processor": platform.processor(),
    }

    # Check for Apple Silicon
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    hardware_info["is_apple_silicon"] = is_apple_silicon

    # Configure GPUs
    gpus = tf.config.list_physical_devices("GPU")
    hardware_info["num_gpus"] = len(gpus)
    hardware_info["gpus"] = [gpu.name for gpu in gpus]

    # Apply Metal-specific optimizations for Apple Silicon
    if is_apple_silicon:
        try:
            # Apply Metal optimizations from memory_utils
            from ..utils.memory_utils import configure_metal_for_stability

            metal_optimized = configure_metal_for_stability()
            hardware_info["metal_optimized"] = metal_optimized

            # Add fallback for graph optimizer issues
            if not metal_optimized:
                os.environ["TF_USE_LEGACY_KERAS"] = (
                    "1"  # Use legacy Keras implementation
                )
                os.environ["TF_METAL_DISABLE_GRAPH_OPTIMIZER"] = (
                    "1"  # Disable problematic optimizer
                )

                hardware_info["metal_optimizer_disabled"] = True
                print(
                    "Disabled Metal graph optimizer due to known compatibility issues"
                )
        except Exception as e:
            print(f"Warning: Error configuring Metal backend: {e}")
            hardware_info["metal_error"] = str(e)

    # Configure CPU threads
    num_threads = config.get("hardware", {}).get("num_threads", 0)
    if num_threads > 0:
        tf.config.threading.set_inter_op_parallelism_threads(num_threads)
        tf.config.threading.set_intra_op_parallelism_threads(num_threads)
        hardware_info["num_threads"] = num_threads

    return hardware_info


def print_hardware_summary():
    """Print a summary of available hardware for TensorFlow"""

    print("\n=== Hardware Summary ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")

    # Check for GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\nGPUs Available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name if hasattr(gpu, 'name') else gpu}")

            # Try to get memory info
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details and "memory_limit" in gpu_details:
                    mem_gb = round(gpu_details["memory_limit"] / (1024**3), 2)
                    print(f"    Memory: {mem_gb} GB")
            except:
                pass
    else:
        print("\nNo GPUs available")

    # Check if using Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("\nApple Silicon detected")
        print("  Metal support is available for TensorFlow acceleration")

    print("\nCPU Information:")
    print(f"  Logical CPUs: {tf.config.threading.get_inter_op_parallelism_threads()}")

    # Check if mixed precision is available
    try:
        policy = tf.keras.mixed_precision.global_policy()
        print(f"\nCurrent precision policy: {policy.name}")
    except:
        print("\nMixed precision status: Unknown")

    print("======================\n")


def get_optimal_batch_size(
    model, starting_batch_size=32, target_memory_usage=0.7, max_attempts=5
):
    """Estimate an optimal batch size for a model based on memory constraints

    This is an experimental function that tries to find a batch size that
    uses a target fraction of available GPU memory.

    Args:
        model: A TensorFlow model
        starting_batch_size: Initial batch size to try
        target_memory_usage: Target fraction of memory to utilize (0.0-1.0)
        max_attempts: Maximum number of batch size adjustments to try

    Returns:
        Estimated optimal batch size or the starting_batch_size if estimation fails
    """
    # Ensure we have GPU available, otherwise return the starting batch size
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU available, using default batch size")
        return starting_batch_size

    # Get input shape from the model
    if hasattr(model, "input_shape"):
        input_shape = model.input_shape
        if isinstance(input_shape, tuple) and None in input_shape:
            # Replace None with a reasonable value (batch dimension)
            input_shape = list(input_shape)
            input_shape[0] = starting_batch_size
            input_shape = tuple(input_shape)
    else:
        print("Could not determine model input shape, using default batch size")
        return starting_batch_size

    try:
        # Try to estimate appropriate batch size
        current_batch_size = starting_batch_size

        for _ in range(max_attempts):
            # Create a test batch
            test_batch = tf.random.normal(input_shape)

            # Run a forward pass
            with tf.GradientTape() as tape:
                _ = model(test_batch, training=True)

            # Try to get memory info
            try:
                memory_info = tf.config.experimental.get_memory_info("GPU:0")
                if memory_info:
                    current_usage = memory_info["current"] / memory_info["peak"]
                    if current_usage < target_memory_usage * 0.8:
                        # Increase batch size
                        current_batch_size = int(current_batch_size * 1.5)
                    elif current_usage > target_memory_usage * 1.1:
                        # Decrease batch size
                        current_batch_size = int(current_batch_size * 0.7)
                    else:
                        # Good batch size found
                        break
            except:
                # If memory info not available, make a conservative guess
                break

        print(f"Estimated optimal batch size: {current_batch_size}")
        return current_batch_size

    except Exception as e:
        print(f"Error estimating optimal batch size: {e}")
        return starting_batch_size
