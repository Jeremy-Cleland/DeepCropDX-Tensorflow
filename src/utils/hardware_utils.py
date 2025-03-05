"""
Hardware configuration utilities for TensorFlow setup.

This module provides functions to configure TensorFlow for optimal 
performance on different hardware platforms (CPU, GPU, Apple Silicon).
"""

import platform
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def configure_hardware(config):
    """Configure TensorFlow for hardware acceleration

    This function configures TensorFlow based on the available hardware and
    the provided configuration. It handles CPU threading, GPU memory growth,
    Apple Silicon Metal support, and mixed precision training.

    Args:
        config: Configuration dictionary with hardware settings

    Returns:
        Dictionary with hardware configuration information
    """
    hardware_info = {
        "platform": platform.system(),
        "cpu_type": platform.machine(),
        "tensorflow_version": tf.__version__,
        "gpu_available": len(tf.config.list_physical_devices("GPU")) > 0,
        "devices_used": [],
    }

    hardware_config = config.get("hardware", {})

    # Set threading parameters FIRST (before any TF operations)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(
            hardware_config.get("inter_op_parallelism", 0)
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            hardware_config.get("intra_op_parallelism", 0)
        )
        print("Threading parameters configured successfully")
    except RuntimeError as e:
        print(f"Warning: Could not set threading parameters: {e}")

    # Detect Apple Silicon
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    hardware_info["is_apple_silicon"] = is_apple_silicon

    # Configure TensorFlow for Metal on Apple Silicon
    if (
        hardware_config.get("use_metal", True)
        and is_apple_silicon
        and hardware_info["gpu_available"]
    ):
        print("Configuring TensorFlow for Metal on Apple Silicon")
        hardware_info["using_metal"] = True

        # Enable Metal
        try:
            gpu_devices = tf.config.list_physical_devices("GPU")
            if gpu_devices:
                tf.config.experimental.set_visible_devices(gpu_devices[0], "GPU")
                hardware_info["devices_used"].append("Metal GPU")

                # Enable memory growth to prevent allocating all GPU memory at once
                if hardware_config.get("memory_growth", True):
                    for gpu in gpu_devices:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                            print(f"Enabled memory growth for {gpu}")
                        except Exception as e:
                            print(
                                f"Warning: Could not set memory growth for {gpu}: {e}"
                            )

                # Use mixed precision if enabled
                if hardware_config.get("mixed_precision", True):
                    try:
                        tf.keras.mixed_precision.set_global_policy("mixed_float16")
                        print("Mixed precision enabled (float16)")
                        hardware_info["mixed_precision"] = True
                    except Exception as e:
                        print(f"Warning: Could not set mixed precision: {e}")
                        hardware_info["mixed_precision"] = False
        except Exception as e:
            print(f"Warning: Error configuring Metal: {e}")
            hardware_info["error_configuring_metal"] = str(e)
            hardware_info["using_metal"] = False

    # For CUDA GPUs
    elif hardware_info["gpu_available"] and hardware_config.get("use_gpu", True):
        print("Configuring TensorFlow for CUDA GPU")
        hardware_info["using_cuda"] = True

        try:
            # Get GPU details
            gpu_devices = tf.config.list_physical_devices("GPU")
            for i, gpu in enumerate(gpu_devices):
                hardware_info["devices_used"].append(
                    f"CUDA GPU {i}: {gpu.name if hasattr(gpu, 'name') else 'Unknown'}"
                )

            # Enable memory growth to prevent allocating all GPU memory at once
            if hardware_config.get("memory_growth", True):
                for gpu in gpu_devices:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"Enabled memory growth for {gpu}")
                    except Exception as e:
                        print(f"Warning: Could not set memory growth for {gpu}: {e}")

            # Use mixed precision if enabled
            if hardware_config.get("mixed_precision", True):
                try:
                    tf.keras.mixed_precision.set_global_policy("mixed_float16")
                    print("Mixed precision enabled (float16)")
                    hardware_info["mixed_precision"] = True
                except Exception as e:
                    print(f"Warning: Could not set mixed precision: {e}")
                    hardware_info["mixed_precision"] = False
        except Exception as e:
            print(f"Warning: Error configuring GPU: {e}")
            hardware_info["error_configuring_gpu"] = str(e)
    else:
        print("Using CPU for computation")
        hardware_info["using_cpu"] = True
        hardware_info["devices_used"].append("CPU")

    # Set memory limit if specified
    memory_limit_mb = hardware_config.get("memory_limit_mb")
    if memory_limit_mb:
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit_mb
                        )
                    ],
                )
            print(f"GPU memory limit set to {memory_limit_mb}MB")
            hardware_info["memory_limit_mb"] = memory_limit_mb
        except Exception as e:
            print(f"Warning: Could not set memory limit: {e}")

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
