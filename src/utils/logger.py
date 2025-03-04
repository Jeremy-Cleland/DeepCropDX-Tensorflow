import os
import logging
import json
import time
import platform
from pathlib import Path
import tensorflow as tf
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np

# Import psutil conditionally for hardware monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from config.config import get_paths


class Logger:
    def __init__(self, name, log_dir=None, config=None, logger_type="training"):
        """Initialize the logging system.

        Args:
            name: Name of the logger (e.g., model name or experiment name)
            log_dir: Directory to save logs. If None, uses the trials directory.
            config: Configuration dictionary for logging settings.
            logger_type: Type of logger - "training" or "evaluation"
        """
        self.name = name
        self.config = config or {}
        self.paths = get_paths()
        self.logger_type = logger_type

        # Set up log directory
        if log_dir is None:
            # If no log_dir is provided, use trials directory
            # This shouldn't happen with our configuration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = self.paths.trials_dir / f"{name}_{timestamp}"
        else:
            # Use the provided directory
            log_dir_path = Path(log_dir)

            # Create subdirectory for logger type
            if logger_type == "training":
                self.log_dir = log_dir_path / "training"
            elif logger_type == "evaluation":
                self.log_dir = log_dir_path / "evaluation"
            else:
                # Default to root of log_dir if type is unknown
                self.log_dir = log_dir_path

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(f"{name}_{logger_type}")
        self.logger.setLevel(self._get_log_level())

        # Clear any existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Set up file handler
        log_file = Path(self.log_dir / f"{name}_{logger_type}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Set up TensorBoard if enabled
        self.tensorboard_writer = None
        if self.config.get("logging", {}).get("tensorboard", True):
            tensorboard_dir = Path(self.log_dir / "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard_writer = tf.summary.create_file_writer(tensorboard_dir)

        # Initialize metrics tracking
        self.metrics = {}
        self.start_time = time.time()

        # Log system info (only for training logger to avoid duplication)
        if logger_type == "training":
            self._log_system_info()

        self.logger.info(
            f"{logger_type.capitalize()} logger initialized. Logs will be saved to {self.log_dir}"
        )

    def _get_log_level(self):
        """Get log level from config or default to INFO"""
        level_str = self.config.get("logging", {}).get("level", "INFO").upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(level_str, logging.INFO)

    def _log_system_info(self):
        """Log information about the system and environment"""
        # System info
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "tensorflow_version": tf.__version__,
            "timestamp": datetime.now().isoformat(),
            "logger_type": self.logger_type,
        }

        # Add more detailed system info if psutil is available
        if PSUTIL_AVAILABLE:
            system_info.update(
                {
                    "cpu_count": psutil.cpu_count(logical=False),
                    "logical_cpus": psutil.cpu_count(logical=True),
                    "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                }
            )

        # Check for GPU/Metal
        system_info["gpu_available"] = len(tf.config.list_physical_devices("GPU")) > 0
        system_info["metal_available"] = (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        )

        # Get GPU details
        devices = []
        for i, gpu in enumerate(tf.config.list_physical_devices("GPU")):
            devices.append(f"GPU {i}: {gpu.name}")

        # For Apple Silicon
        if system_info["metal_available"]:
            devices.append("Metal: Apple Silicon")

        system_info["devices"] = devices

        # Get more detailed GPU info if available
        try:
            for i, gpu in enumerate(tf.config.list_physical_devices("GPU")):
                # Try to get memory limit if set
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if gpu_details and "memory_limit" in gpu_details:
                        mem_gb = round(gpu_details["memory_limit"] / (1024**3), 2)
                        system_info[f"gpu_{i}_memory_limit_gb"] = mem_gb
                except:
                    pass
        except:
            pass

        # Log to file and console
        self.logger.info(f"System Info: {json.dumps(system_info, indent=2)}")

        # Save as JSON
        system_info_path = Path(self.log_dir / "system_info.json")
        with open(system_info_path, "w") as f:
            json.dump(system_info, f, indent=4)

    def log_info(self, message):
        """Log an info message"""
        self.logger.info(message)

    def log_warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)

    def log_error(self, message):
        """Log an error message"""
        self.logger.error(message)

    def log_debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)

    def log_config(self, config):
        """Log the configuration used for training"""
        # Create a clean copy of the config that's JSON-serializable
        clean_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                clean_config[key] = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                        clean_config[key][k] = v
            elif isinstance(value, (str, int, float, bool, list)) or value is None:
                clean_config[key] = value

        self.logger.info(f"Configuration: {json.dumps(clean_config, indent=2)}")

        # Save config as JSON
        config_path = Path(self.log_dir / "config.json")
        with open(config_path, "w") as f:
            json.dump(clean_config, f, indent=4)

    def log_model_summary(self, model):
        """Log model architecture summary"""
        # Create a string buffer to capture the summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))

        # Log to file
        summary_path = Path(self.log_dir / "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(model_summary) + "\n")

        self.logger.info(f"Model summary saved to {summary_path}")

        # Try to generate a model diagram
        try:
            if len(model.layers) <= 100:  # Skip for very complex models
                dot_img_file = Path(self.log_dir / "model_diagram.png")
                tf.keras.utils.plot_model(
                    model, to_file=dot_img_file, show_shapes=True, show_layer_names=True
                )
                self.logger.info(f"Model diagram saved to {dot_img_file}")
        except Exception as e:
            self.logger.debug(f"Could not generate model diagram: {e}")

    def log_metrics(self, metrics, step=None):
        """Log metrics during training or evaluation"""
        # Update metrics dictionary
        for key, value in metrics.items():
            # Convert TensorFlow tensors to Python types
            if hasattr(value, "numpy"):
                try:
                    value = value.numpy()
                except:
                    pass

            # Convert NumPy values to Python types
            if hasattr(np, "float32") and isinstance(
                value, (np.float32, np.float64, np.int32, np.int64)
            ):
                value = value.item()

            self.metrics[key] = value

        # Log to console and file
        metrics_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
        )
        self.logger.info(f"Metrics - {metrics_str}")

        # Log to TensorBoard if enabled
        if self.tensorboard_writer and step is not None:
            with self.tensorboard_writer.as_default():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        tf.summary.scalar(key, value, step=step)
                self.tensorboard_writer.flush()

    def log_hardware_metrics(self, step=None):
        """Log hardware utilization metrics including GPU/Metal activity detection"""
        if not PSUTIL_AVAILABLE:
            self.logger.warning(
                "psutil not installed; hardware metrics logging disabled"
            )
            return {}

        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)

            # GPU usage tracking
            gpu_info = {}
            gpu_active = False

            # Check if GPU devices are available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                gpu_info["available"] = len(gpus)

                # Try to detect if GPU is being used
                try:
                    # For Metal on Apple Silicon
                    if platform.system() == "Darwin" and platform.machine() == "arm64":
                        # Simple test tensor operation on GPU to check if it's working
                        with tf.device("/GPU:0"):
                            # Create and immediately use a test tensor
                            a = tf.random.normal([1000, 1000])
                            b = tf.random.normal([1000, 1000])
                            c = tf.matmul(a, b)  # Matrix multiplication to engage GPU
                            # Force execution
                            _ = c.numpy()

                            # Check if operation was actually done on GPU
                            gpu_active = True
                            gpu_info["active"] = True

                        # Try to get memory usage (experimental API)
                        try:
                            memory_info = tf.config.experimental.get_memory_info(
                                "/device:GPU:0"
                            )
                            if memory_info:
                                gpu_info["memory_used_mb"] = memory_info.get(
                                    "current", 0
                                ) / (1024**2)
                                gpu_info["memory_peak_mb"] = memory_info.get(
                                    "peak", 0
                                ) / (1024**2)
                        except:
                            pass
                except Exception as e:
                    self.logger.debug(f"Error checking GPU activity: {e}")
                    gpu_info["error"] = str(e)

            # Format hardware metrics string
            hw_metrics_str = f"Hardware - CPU: {cpu_percent}%, Memory: {memory_percent}% ({memory_used_gb:.2f} GB)"

            # Add GPU info if available
            if gpus:
                gpu_status = "ACTIVE" if gpu_active else "INACTIVE"
                hw_metrics_str += f", GPU: {gpu_status}"

                # Add memory info if available
                if "memory_used_mb" in gpu_info:
                    hw_metrics_str += f" (Memory: {gpu_info['memory_used_mb']:.2f} MB)"

            # Log to console and file
            self.logger.info(hw_metrics_str)

            # Create hardware metrics dict for tensorboard
            hw_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory_used_gb,
                "gpu_active": 1 if gpu_active else 0,
            }

            # Add per-CPU metrics
            for i, cpu in enumerate(per_cpu):
                hw_metrics[f"cpu_{i}_percent"] = cpu

            # Add GPU metrics
            if "memory_used_mb" in gpu_info:
                hw_metrics["gpu_memory_used_mb"] = gpu_info["memory_used_mb"]
                hw_metrics["gpu_memory_peak_mb"] = gpu_info.get("memory_peak_mb", 0)

            # Log to TensorBoard if enabled
            if self.tensorboard_writer and step is not None:
                with self.tensorboard_writer.as_default():
                    for key, value in hw_metrics.items():
                        tf.summary.scalar(f"hardware/{key}", value, step=step)
                    self.tensorboard_writer.flush()

            return hw_metrics

        except Exception as e:
            self.logger.warning(f"Error in hardware metrics logging: {str(e)}")
            return {}

    def log_training_progress(self, epoch, batch, metrics, total_batches):
        """Log progress during training with tqdm progress bar"""
        # Calculate time and ETA
        time_elapsed = time.time() - self.start_time
        progress = (batch + 1) / total_batches
        eta = time_elapsed / (progress + 1e-8) * (1 - progress)

        # Format metrics for logging
        metrics_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
        )

        # Log to file
        self.logger.info(
            f"Epoch {epoch+1} - Batch {batch+1}/{total_batches} - {metrics_str} - "
            f"ETA: {eta:.2f}s"
        )

        # For tqdm integration, we would typically use the ProgressBarCallback class
        # which is managed by the Trainer class

    def log_images(self, images, step, name="images"):
        """Log images to TensorBoard

        Args:
            images: Batch of images (shape [N, H, W, C])
            step: Current step
            name: Name for the images
        """
        if self.tensorboard_writer:
            with self.tensorboard_writer.as_default():
                tf.summary.image(name, images, step=step, max_outputs=10)
                self.tensorboard_writer.flush()

    def log_confusion_matrix(self, cm, class_names, step):
        """Log confusion matrix as an image to TensorBoard

        Args:
            cm: Confusion matrix (shape [num_classes, num_classes])
            class_names: List of class names
            step: Current step
        """
        if self.tensorboard_writer:
            try:
                import matplotlib.pyplot as plt
                import io

                # Create figure and plot confusion matrix
                figure = plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title("Confusion Matrix")
                plt.colorbar()
                tick_marks = range(len(class_names))
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)

                # Label the matrix
                thresh = cm.max() / 2.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j,
                            i,
                            format(cm[i, j], "d"),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                        )

                plt.tight_layout()
                plt.ylabel("True label")
                plt.xlabel("Predicted label")

                # Convert figure to image
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close(figure)
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)

                # Log to TensorBoard
                with self.tensorboard_writer.as_default():
                    tf.summary.image("confusion_matrix", image, step=step)
                    self.tensorboard_writer.flush()

                # Also save the confusion matrix as an image file
                cm_path = Path(self.log_dir / f"confusion_matrix_epoch_{step}.png")
                plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title("Confusion Matrix")
                plt.colorbar()
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)

                # Label the matrix
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j,
                            i,
                            format(cm[i, j], "d"),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                        )

                plt.tight_layout()
                plt.ylabel("True label")
                plt.xlabel("Predicted label")
                plt.savefig(cm_path)
                plt.close()

                self.logger.info(f"Confusion matrix saved to {cm_path}")

            except Exception as e:
                self.logger.warning(f"Error logging confusion matrix: {e}")

    def save_final_metrics(self, metrics):
        """Save final metrics at the end of training or evaluation"""
        # Update metrics dictionary
        for key, value in metrics.items():
            # Convert TensorFlow tensors to Python types
            if hasattr(value, "numpy"):
                try:
                    value = value.numpy()
                    if hasattr(value, "item"):
                        value = value.item()
                except:
                    value = str(value)

            # Convert NumPy values to Python types
            if hasattr(np, "float32") and isinstance(
                value, (np.float32, np.float64, np.int32, np.int64)
            ):
                value = value.item()

            self.metrics[key] = value

        # Add timing information
        self.metrics["training_time_seconds"] = time.time() - self.start_time
        self.metrics["training_time_human"] = self._format_time(
            self.metrics["training_time_seconds"]
        )
        self.metrics["timestamp_end"] = datetime.now().isoformat()
        self.metrics["logger_type"] = self.logger_type

        # First save to the log directory
        log_metrics_path = Path(self.log_dir / f"final_metrics_{self.logger_type}.json")
        with open(log_metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        # Also save to the parent directory (for the model registry)
        # Only if this is a training logger
        if self.logger_type == "training":
            parent_metrics_path = Path(self.log_dir).parent / "final_metrics.json"
            with open(parent_metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=4)
            self.logger.info(
                f"Final metrics saved to {log_metrics_path} and {parent_metrics_path}"
            )
        else:
            self.logger.info(f"Final metrics saved to {log_metrics_path}")

        return self.metrics

    def _format_time(self, seconds):
        """Format time in seconds to a human-readable string"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if hours > 0:
            parts.append(f"{int(hours)}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{int(minutes)}m")
        parts.append(f"{int(seconds)}s")

        return " ".join(parts)


class ProgressBarManager:
    """A class to manage tqdm progress bars for training and evaluation"""

    def __init__(self, total=None, desc=None, position=0, leave=True):
        """Initialize a progress bar manager

        Args:
            total: Total number of items
            desc: Description for the progress bar
            position: Position of the progress bar (for nested bars)
            leave: Whether to leave the progress bar after completion
        """
        self.total = total
        self.desc = desc
        self.position = position
        self.leave = leave
        self.pbar = None

    def __enter__(self):
        """Create and return the progress bar when entering a context"""
        self.pbar = tqdm(
            total=self.total, desc=self.desc, position=self.position, leave=self.leave
        )
        return self.pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the progress bar when exiting the context"""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
