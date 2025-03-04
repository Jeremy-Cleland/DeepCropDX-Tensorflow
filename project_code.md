# Project Code Overview

*Generated on 2025-03-04 14:17:58*

## Table of Contents

- [compile.py](#compile-py)
- [deepcropdx.yml](#deepcropdx-yml)
- [main.py](#main-py)
- [setup.py](#setup-py)
- [src/config/__init__.py](#src-config-__init__-py)
- [src/config/config.py](#src-config-config-py)
- [src/config/config.yaml](#src-config-config-yaml)
- [src/config/config_loader.py](#src-config-config_loader-py)
- [src/config/model_configs/__init__.py](#src-config-model_configs-__init__-py)
- [src/config/model_configs/models.yaml](#src-config-model_configs-models-yaml)
- [src/evaluation/__init__.py](#src-evaluation-__init__-py)
- [src/evaluation/metrics.py](#src-evaluation-metrics-py)
- [src/evaluation/visualization.py](#src-evaluation-visualization-py)
- [src/model_registry/__init__.py](#src-model_registry-__init__-py)
- [src/model_registry/registry_manager.py](#src-model_registry-registry_manager-py)
- [src/models/__init__.py](#src-models-__init__-py)
- [src/models/model_factory.py](#src-models-model_factory-py)
- [src/preprocessing/__init__.py](#src-preprocessing-__init__-py)
- [src/preprocessing/data_loader.py](#src-preprocessing-data_loader-py)
- [src/scripts/__init__.py](#src-scripts-__init__-py)
- [src/scripts/evaluate.py](#src-scripts-evaluate-py)
- [src/scripts/registry_cli.py](#src-scripts-registry_cli-py)
- [src/scripts/train.py](#src-scripts-train-py)
- [src/training/__init__.py](#src-training-__init__-py)
- [src/training/trainer.py](#src-training-trainer-py)
- [src/utils/__init__.py](#src-utils-__init__-py)
- [src/utils/logger.py](#src-utils-logger-py)
- [src/utils/report_generator.py](#src-utils-report_generator-py)

## Code Files

### compile.py

```python
#!/usr/bin/env python3
"""
Script to compile Python and YAML files in a project into a single markdown document.
This is useful for sharing code with LLMs for analysis.
"""

import os
import argparse
from datetime import datetime


def find_files(directory, extensions=[".py", ".yaml", ".yml"], ignore_dirs=None):
    """Find all files with specified extensions in the given directory and its subdirectories."""
    if ignore_dirs is None:
        ignore_dirs = [
            ".git",
            ".github",
            "__pycache__",
            "venv",
            "env",
            ".venv",
            ".env",
            "build",
            "dist",
        ]

    matching_files = []

    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                matching_files.append(os.path.join(root, file))

    return sorted(matching_files)


def get_relative_path(file_path, base_dir):
    """Get the path relative to the base directory."""
    return os.path.relpath(file_path, base_dir)


def get_file_language(file_path):
    """Determine the language based on file extension."""
    if file_path.endswith((".yaml", ".yml")):
        return "yaml"
    elif file_path.endswith(".py"):
        return "python"
    else:
        return "text"


def create_markdown(files, base_dir, output_file):
    """Create a markdown document from the files."""
    # Count file types
    python_files = [f for f in files if f.endswith(".py")]
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Project Code Overview\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("## Table of Contents\n\n")

        # Generate table of contents
        for file_path in files:
            rel_path = get_relative_path(file_path, base_dir)
            anchor = rel_path.replace("/", "-").replace(".", "-").replace(" ", "-")
            f.write(f"- [{rel_path}](#{anchor})\n")

        f.write("\n## Code Files\n\n")

        # Write each file with its content
        for file_path in files:
            rel_path = get_relative_path(file_path, base_dir)
            anchor = rel_path.replace("/", "-").replace(".", "-").replace(" ", "-")
            language = get_file_language(file_path)

            f.write(f"### {rel_path}\n\n")
            f.write(f"```{language}\n")

            try:
                with open(file_path, "r", encoding="utf-8") as code_file:
                    f.write(code_file.read())
            except UnicodeDecodeError:
                try:
                    # Try with a different encoding if UTF-8 fails
                    with open(file_path, "r", encoding="latin-1") as code_file:
                        f.write(code_file.read())
                except Exception as e:
                    f.write(f"# Error reading file: {str(e)}\n")
            except Exception as e:
                f.write(f"# Error reading file: {str(e)}\n")

            f.write("```\n\n")
            f.write("---\n\n")

        # Add summary information
        f.write(f"## Summary\n\n")
        f.write(f"Total files: {len(files)}\n")
        f.write(f"- Python files: {len(python_files)}\n")
        f.write(f"- YAML files: {len(yaml_files)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compile Python and YAML files into a markdown document."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Directory to scan for files (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="project_code.md",
        help="Output markdown file (default: project_code.md)",
    )
    parser.add_argument(
        "--ignore",
        "-i",
        type=str,
        nargs="+",
        default=[
            ".git",
            ".github",
            "__pycache__",
            "venv",
            "env",
            ".venv",
            ".env",
            "build",
            "dist",
        ],
        help="Directories to ignore (default: .git, __pycache__, venv, etc.)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        type=str,
        nargs="+",
        default=[".py", ".yaml", ".yml"],
        help="File extensions to include (default: .py, .yaml, .yml)",
    )

    args = parser.parse_args()

    base_dir = os.path.abspath(args.directory)
    files = find_files(base_dir, args.extensions, args.ignore)

    if not files:
        print(f"No matching files found in {base_dir}")
        return

    create_markdown(files, base_dir, args.output)
    print(f"Markdown document created at {args.output}")
    print(f"Found {len(files)} files")

    # Print breakdown by type
    python_files = [f for f in files if f.endswith(".py")]
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]
    print(f"- {len(python_files)} Python files")
    print(f"- {len(yaml_files)} YAML files")


if __name__ == "__main__":
    main()
```

---

### deepcropdx.yml

```yaml
name: deepcropdx
channels:
  - conda-forge
  - apple
  - defaults
dependencies:
  - python=3.10 
  - pip
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scipy
  - statsmodels
  - scikit-learn
  - h5py
  - jupyterlab
  - pillow
  - grpcio
  - protobuf
  - typing-extensions
  - six
  - traitlets
  - tornado
  - sympy
  - mpmath
  - numexpr
  - networkx
  - cmake
  - numba
  - dask
  - xarray
  - tqdm
  - psutil
  - pytest
  - plotly
  - altair
  - black
  - flake8
  - pylint
  - jupytext
  - pytorch
  - torchvision
  - torchaudio
  - autograd
  - category_encoders
  - fancyimpute
  - ipywidgets
  - flask
  - jinja2
  - optuna
  - pyyaml
  - pip:
      - tensorflow-macos
      - tensorflow-metal
      - tensorboard
      - tensorflow-addons
      - tensorflow-probability
      - absl-py
      - opt-einsum
      - rich
      - termcolor
      - tfds-nightly
      - gin-config
      - opencv-python-headless
      - np
      - pymc3
      - pgmpy
      - keras
```

---

### main.py

```python
import os
import argparse
import yaml
import tensorflow as tf
from pathlib import Path
from tqdm.auto import tqdm
import time
from datetime import datetime

from src.config.config import get_paths
from src.config.config_loader import ConfigLoader
from src.preprocessing.data_loader import DataLoader
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.report_generator import ReportGenerator
from src.utils.logger import Logger
from src.model_registry.registry_manager import ModelRegistryManager


def configure_hardware(config):
    """Configure TensorFlow for hardware acceleration

    Args:
        config: Configuration dictionary with hardware settings
    """
    hardware_config = config.get("hardware", {})

    # Configure TensorFlow for Metal on Apple Silicon
    if (
        hardware_config.get("use_metal", True)
        and hasattr(tf.test, "is_built_with_metal")
        and tf.config.list_physical_devices("GPU")
    ):
        print("Configuring TensorFlow for Metal on Apple Silicon")

        # Enable Metal
        try:
            tf.config.experimental.set_visible_devices(
                tf.config.list_physical_devices("GPU")[0], "GPU"
            )

            # Enable memory growth to prevent allocating all GPU memory at once
            if hardware_config.get("memory_growth", True):
                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)

            # Use mixed precision if enabled
            if hardware_config.get("mixed_precision", True):
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision enabled (float16)")
        except Exception as e:
            print(f"Warning: Error configuring Metal: {e}")

    # For CUDA GPUs
    elif tf.config.list_physical_devices("GPU") and hardware_config.get(
        "use_gpu", True
    ):
        print("Configuring TensorFlow for CUDA GPU")

        try:
            # Enable memory growth to prevent allocating all GPU memory at once
            if hardware_config.get("memory_growth", True):
                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)

            # Use mixed precision if enabled
            if hardware_config.get("mixed_precision", True):
                tf.keras.mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision enabled (float16)")
        except Exception as e:
            print(f"Warning: Error configuring GPU: {e}")
    else:
        print("Using CPU for computation")

    # Configure other TensorFlow options
    tf.config.threading.set_inter_op_parallelism_threads(
        hardware_config.get("inter_op_parallelism", 0)
    )
    tf.config.threading.set_intra_op_parallelism_threads(
        hardware_config.get("intra_op_parallelism", 0)
    )


def train_model(
    model_name,
    config,
    data_loader,
    model_factory,
    train_data,
    val_data,
    test_data,
    class_names,
    batch_logger=None,
):
    """Train a single model and return results

    Args:
        model_name: Name of the model to train
        config: Configuration dictionary
        data_loader: DataLoader instance
        model_factory: ModelFactory instance
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset
        class_names: Dictionary of class names
        batch_logger: Optional logger for batch training process

    Returns:
        Tuple of (success_flag, metrics_dict)
    """
    print(f"\n{'='*80}")
    print(f"Training model: {model_name}")
    print(f"{'='*80}")

    if batch_logger:
        batch_logger.log_info(f"Starting training for model: {model_name}")

    try:
        # Get model hyperparameters (combining defaults with model-specific)
        config_loader = ConfigLoader()
        hyperparams = config_loader.get_hyperparameters(model_name, config)

        # Update config with these hyperparameters
        training_config = config.get("training", {}).copy()
        training_config.update(hyperparams)
        config["training"] = training_config

        print(f"Training configuration for {model_name}:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")

        if batch_logger:
            batch_logger.log_info(
                f"Hyperparameters for {model_name}: {training_config}"
            )

        # Create model
        model = model_factory.get_model(model_name, num_classes=len(class_names))

        # Train model
        trainer = Trainer(config)
        model, history, metrics = trainer.train(
            model, model_name, train_data, val_data, test_data
        )

        # Generate report
        if config.get("reporting", {}).get("generate_html_report", True):
            report_generator = ReportGenerator(config)
            run_dir = metrics.get("run_dir", "")
            report_path = report_generator.generate_model_report(
                model_name, run_dir, metrics, history, class_names
            )
            print(f"Report generated at {report_path}")

            if batch_logger:
                batch_logger.log_info(
                    f"Report for {model_name} generated at {report_path}"
                )

        print(f"Training completed for {model_name}")
        accuracy = metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
        training_time = metrics.get("training_time_seconds", 0)

        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} seconds")

        if batch_logger:
            batch_logger.log_info(f"Model {model_name} training successful")
            batch_logger.log_info(f"Final accuracy: {accuracy:.4f}")
            batch_logger.log_info(f"Training time: {training_time:.2f} seconds")

            # Log summary metrics for the batch report
            batch_summary = {
                f"{model_name}_accuracy": accuracy,
                f"{model_name}_training_time": training_time,
            }
            batch_logger.log_metrics(batch_summary)

        return True, metrics

    except Exception as e:
        error_msg = f"Error training model {model_name}: {e}"
        print(error_msg)
        import traceback

        trace = traceback.format_exc()
        print(trace)

        if batch_logger:
            batch_logger.log_error(error_msg)
            batch_logger.log_error(trace)

        return False, {"error": str(e)}


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Plant Disease Detection Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Model architectures to train (space-separated list)",
    )
    parser.add_argument(
        "--all_models",
        action="store_true",
        help="Train all models defined in the configuration",
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to the dataset directory"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size for training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs for training",
    )
    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    # Get project paths
    paths = get_paths()

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()

    # Set project information
    project_info = config.get("project", {})
    project_name = project_info.get("name", "Plant Disease Detection")
    project_version = project_info.get("version", "1.0.0")

    print(f"Starting {project_name} v{project_version}")

    # Create a directory for batch logs
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_dir = paths.logs_dir / f"batch_{batch_timestamp}"
    batch_log_dir.mkdir(parents=True, exist_ok=True)

    # Initialize batch logger for overall process tracking
    batch_logger = Logger(
        "batch_training",
        log_dir=batch_log_dir,
        config=config.get("logging", {}),
        logger_type="batch",
    )

    batch_logger.log_info(f"Starting {project_name} v{project_version}")
    batch_logger.log_config(config)

    # Override configuration with command-line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
        print(f"Overriding epochs: {args.epochs}")
        batch_logger.log_info(f"Overriding epochs: {args.epochs}")
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
        print(f"Overriding batch size: {args.batch_size}")
        batch_logger.log_info(f"Overriding batch size: {args.batch_size}")

    # Configure hardware
    configure_hardware(config)
    batch_logger.log_hardware_metrics(step=0)

    # Log system information
    batch_logger.log_info("Loading datasets...")

    # Load data (only once for all models)
    data_loader = DataLoader(config)
    train_data, val_data, test_data, class_names = data_loader.load_data(args.data_dir)

    batch_logger.log_info(f"Datasets loaded with {len(class_names)} classes")
    batch_logger.log_info(f"Classes: {list(class_names.values())}")

    # Create model factory
    model_factory = ModelFactory()

    # Initialize model registry
    registry = ModelRegistryManager()

    # Determine which models to train
    models_to_train = []

    if args.all_models:
        # Train all models in the configuration
        models_to_train = config_loader.get_all_model_names()
        print(f"Will train all {len(models_to_train)} models from configuration")
        batch_logger.log_info(
            f"Will train all {len(models_to_train)} models from configuration: {', '.join(models_to_train)}"
        )
    elif args.models:
        # Train specific models
        models_to_train = args.models
        msg = f"Will train {len(models_to_train)} specified models: {', '.join(models_to_train)}"
        print(msg)
        batch_logger.log_info(msg)
    else:
        # Default to training a single model (ResNet50)
        models_to_train = ["ResNet50"]
        print("No models specified, defaulting to ResNet50")
        batch_logger.log_info("No models specified, defaulting to ResNet50")

    # Train all specified models
    results = {}
    successful_models = 0
    failed_models = 0

    for model_name in (model_pbar := tqdm(models_to_train, desc="Models", position=0)):
        model_pbar.set_description(f"Training {model_name}")

        model_start_time = time.time()
        success, metrics = train_model(
            model_name,
            config,
            data_loader,
            model_factory,
            train_data,
            val_data,
            test_data,
            class_names,
            batch_logger,
        )
        model_time = time.time() - model_start_time

        results[model_name] = metrics
        if success:
            successful_models += 1
        else:
            failed_models += 1

        status_str = f"{'✓' if success else '✗'} in {model_time:.1f}s"
        model_pbar.set_postfix_str(status_str)

        # Log model completion
        accuracy = (
            metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
            if "error" not in metrics
            else 0
        )
        batch_logger.log_info(
            f"Model {model_name} completed - Status: {'Success' if success else 'Failed'}, Time: {model_time:.2f}s, Accuracy: {accuracy:.4f}"
        )

    # Generate comparison report if multiple models were trained
    if len(results) > 1 and config.get("reporting", {}).get(
        "generate_html_report", True
    ):
        try:
            comparison_data = []
            for model_name, metrics in results.items():
                if "error" not in metrics:
                    comparison_data.append({"name": model_name, "metrics": metrics})

            if comparison_data:
                report_generator = ReportGenerator(config)
                comparison_path = report_generator.generate_comparison_report(
                    comparison_data
                )
                print(f"Model comparison report generated at {comparison_path}")
                batch_logger.log_info(
                    f"Model comparison report generated at {comparison_path}"
                )
        except Exception as e:
            error_msg = f"Error generating comparison report: {e}"
            print(error_msg)
            batch_logger.log_error(error_msg)

    # Print summary of all trained models
    total_time = time.time() - start_time

    print("\n\nTraining Summary:")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Models trained: {successful_models} successful, {failed_models} failed")

    batch_logger.log_info("\nTraining Summary:")
    batch_logger.log_info(
        f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
    )
    batch_logger.log_info(
        f"Models trained: {successful_models} successful, {failed_models} failed"
    )

    # Prepare the final metrics for the batch logger
    batch_metrics = {
        "total_time_seconds": total_time,
        "total_time_minutes": total_time / 60,
        "successful_models": successful_models,
        "failed_models": failed_models,
        "total_models": len(models_to_train),
    }

    # Create detailed model results for batch logging
    for model_name, metrics in results.items():
        if "error" in metrics:
            print(f"{model_name}: Failed - {metrics['error']}")
            batch_logger.log_info(f"{model_name}: Failed - {metrics['error']}")
            batch_metrics[f"{model_name}_status"] = "failed"
            batch_metrics[f"{model_name}_error"] = metrics["error"]
        else:
            accuracy = metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
            train_time = metrics.get("training_time_seconds", 0)
            print(
                f"{model_name}: Success - Accuracy: {accuracy:.4f} - Time: {train_time:.2f}s"
            )
            batch_logger.log_info(
                f"{model_name}: Success - Accuracy: {accuracy:.4f} - Time: {train_time:.2f}s"
            )
            batch_metrics[f"{model_name}_status"] = "success"
            batch_metrics[f"{model_name}_accuracy"] = accuracy
            batch_metrics[f"{model_name}_training_time"] = train_time

    # Save final batch metrics
    batch_logger.save_final_metrics(batch_metrics)

    print("=" * 80)
    print("Training completed.")
    batch_logger.log_info("Training completed.")


if __name__ == "__main__":
    main()
```

---

### setup.py

```python
from setuptools import setup, find_packages

setup(
    name="deepcropdx",
    version="1.0.0",
    description="Deep Learning Models for Plant Disease Detection",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.7.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "pillow>=8.2.0",
        "jinja2>=3.0.0",
        "pyyaml>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "deepcropdx-train=src.scripts.train:main",
            "deepcropdx-evaluate=src.scripts.evaluate:main",
            "deepcropdx-registry=src.scripts.registry_cli:main",
        ],
    },
    python_requires=">=3.8",
)
```

---

### src/config/__init__.py

```python
```

---

### src/config/config.py

```python
import os
from pathlib import Path


class ProjectPaths:
    def __init__(self, base_dir=None):
        """Initialize project paths.

        Args:
            base_dir: Base directory of the project. If None, uses the parent directory of this file.
        """
        if base_dir is None:
            # Get the absolute path of the parent directory
            self.base_dir = Path(__file__).parent.parent.parent.absolute()
        else:
            self.base_dir = Path(base_dir).absolute()

        # Source code directories
        self.src_dir = self.base_dir / "src"
        self.config_dir = self.src_dir / "config"
        self.model_configs_dir = self.config_dir / "model_configs"
        self.models_dir = self.src_dir / "models"
        self.preprocessing_dir = self.src_dir / "preprocessing"
        self.evaluation_dir = self.src_dir / "evaluation"
        self.training_dir = self.src_dir / "training"
        self.utils_dir = self.src_dir / "utils"
        self.scripts_dir = self.src_dir / "scripts"

        # Data directories
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"

        # Model output directories
        self.trials_dir = self.base_dir / "trials"

        # Logs directory
        self.logs_dir = self.base_dir / "logs"

        # Ensure critical directories exist
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """Create all necessary directories if they don't exist"""
        directories = [
            self.src_dir,
            self.config_dir,
            self.model_configs_dir,
            self.models_dir,
            self.preprocessing_dir,
            self.evaluation_dir,
            self.training_dir,
            self.utils_dir,
            self.scripts_dir,
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.trials_dir,
            self.logs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_trial_dir(self, model_name, run_id=None):
        """Get the trial directory for a specific model.

        Args:
            model_name: Name of the model (e.g., "EfficientNetB1")
            run_id: Specific run ID. If None, will use a timestamp.

        Returns:
            Path to the model trial directory
        """
        from datetime import datetime

        model_dir = self.trials_dir / model_name

        if run_id is None:
            # Generate a timestamped run ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Find the latest run number
            existing_runs = [
                d for d in model_dir.glob(f"run_{timestamp}_*") if d.is_dir()
            ]
            if existing_runs:
                latest_num = max([int(d.name.split("_")[-1]) for d in existing_runs])
                run_id = f"run_{timestamp}_{(latest_num + 1):03d}"
            else:
                run_id = f"run_{timestamp}_001"

        run_dir = model_dir / run_id

        # Create subdirectories for training and evaluation
        train_dir = run_dir / "training"
        eval_dir = run_dir / "evaluation"
        checkpoints_dir = train_dir / "checkpoints"
        plots_dir = train_dir / "plots"
        tensorboard_dir = train_dir / "tensorboard"

        # Create all directories
        for directory in [
            run_dir,
            train_dir,
            eval_dir,
            checkpoints_dir,
            plots_dir,
            tensorboard_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        return run_dir

    def get_config_path(self):
        """Get the path to the main configuration file"""
        return self.config_dir / "config.yaml"

    def get_model_config_path(self, model_name=None):
        """Get the path to the model configuration file.

        Args:
            model_name: Name of the model. If None, returns the models.yaml path.

        Returns:
            Path to the model configuration file
        """
        if model_name is None:
            return self.model_configs_dir / "models.yaml"

        # Try model-specific file first
        model_file = f"{model_name.lower().split('_')[0]}.yaml"
        specific_path = self.model_configs_dir / model_file

        if specific_path.exists():
            return specific_path

        # Fall back to models.yaml
        return self.model_configs_dir / "models.yaml"


# Create a singleton instance
project_paths = ProjectPaths()


def get_paths():
    """Get the project paths singleton instance"""
    return project_paths
```

---

### src/config/config.yaml

```yaml
project:
  name: Plant Disease Detection
  description: Deep learning models for detecting diseases in plants
  version: 1.0.0

paths:
  data: 
    raw: data/raw
    processed: data/processed
  models: model_registry
  logs: trials

training:
  batch_size: 128
  epochs: 50
  learning_rate: 0.001
  optimizer: adam
  loss: categorical_crossentropy
  metrics: [accuracy, AUC, Precision, Recall]
  early_stopping:
    enabled: true
    patience: 10
    monitor: val_loss
  validation_split: 0.2
  test_split: 0.1
  progress_bar: true

hardware:
  use_metal: true
  mixed_precision: true
  memory_growth: true
  num_parallel_calls: 16
  prefetch_buffer_size: 8

logging:
  level: INFO
  tensorboard: true
  separate_loggers: true

reporting:
  generate_plots: true
  save_confusion_matrix: true
  save_roc_curves: true
  save_precision_recall: true
  generate_html_report: true

data_augmentation:
  rotation_range: 20
  width_shift_range: 0.2
  height_shift_range: 0.2
  shear_range: 0.2
  zoom_range: 0.2
  horizontal_flip: true
  vertical_flip: false
  fill_mode: "nearest"```

---

### src/config/config_loader.py

```python
import os
import yaml
from pathlib import Path

from config.config import get_paths


class ConfigLoader:
    def __init__(self, config_path=None):
        """Initialize the configuration loader with an optional custom config path

        Args:
            config_path: Path to the custom configuration file. If None, uses the default.
        """
        self.paths = get_paths()

        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.paths.get_config_path()

    def get_config(self):
        """Load and return the main configuration

        Returns:
            Dictionary with configuration values, or empty dict if file not found
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                print(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                print(f"Error loading configuration from {self.config_path}: {e}")
                return {}
        else:
            print(f"Configuration file not found at {self.config_path}")
            return {}

    def get_model_config(self, model_name):
        """
        Get configuration for a specific model.
        First checks models.yaml for all models, then falls back to individual files.

        Args:
            model_name: Name of the model to get configuration for

        Returns:
            Dictionary with model configuration

        Raises:
            ValueError: If configuration for the model is not found
        """
        # First try to get from models.yaml (centralized configs)
        models_yaml_path = self.paths.get_model_config_path()
        if models_yaml_path.exists():
            try:
                with open(models_yaml_path, "r") as f:
                    all_configs = yaml.safe_load(f)
                    if all_configs and model_name in all_configs:
                        print(
                            f"Found configuration for {model_name} in {models_yaml_path}"
                        )
                        return all_configs
            except Exception as e:
                print(
                    f"Error loading model configurations from {models_yaml_path}: {e}"
                )

        # Otherwise try model-specific file
        model_config_path = self.paths.get_model_config_path(model_name)

        # If model-specific file exists, load it
        if model_config_path.exists():
            try:
                with open(model_config_path, "r") as f:
                    model_config = yaml.safe_load(f)
                    print(
                        f"Found configuration for {model_name} in {model_config_path}"
                    )
                    return model_config
            except Exception as e:
                print(
                    f"Error loading model configuration from {model_config_path}: {e}"
                )

        # If no config found, raise an error
        raise ValueError(f"Configuration for model {model_name} not found")

    def get_all_model_names(self):
        """Get a list of all available model names from the configuration

        Returns:
            List of model names
        """
        models_yaml_path = self.paths.get_model_config_path()
        if models_yaml_path.exists():
            try:
                with open(models_yaml_path, "r") as f:
                    all_configs = yaml.safe_load(f)
                    if all_configs:
                        return list(all_configs.keys())
            except Exception as e:
                print(f"Error loading model names from {models_yaml_path}: {e}")

        return []

    def get_hyperparameters(self, model_name=None, default_config=None):
        """Get hyperparameters for training, combining default and model-specific configs

        Args:
            model_name: Name of the model to get hyperparameters for (optional)
            default_config: Default configuration to use (optional)

        Returns:
            Dictionary with hyperparameters
        """
        # Start with default config if provided, otherwise load from file
        config = default_config if default_config else self.get_config()

        # Extract training hyperparameters from main config
        hyperparams = config.get("training", {}).copy()

        # If model_name is provided, try to get model-specific hyperparameters
        if model_name:
            try:
                model_config = self.get_model_config(model_name)
                model_hyperparams = model_config.get(model_name, {}).get(
                    "hyperparameters", {}
                )

                # Merge model-specific hyperparameters (they take precedence)
                hyperparams.update(model_hyperparams)
            except Exception as e:
                print(
                    f"Warning: Could not load model-specific hyperparameters for {model_name}: {e}"
                )

        return hyperparams

    def save_config(self, config, output_path=None):
        """Save configuration to a file

        Args:
            config: Configuration dictionary to save
            output_path: Path to save the configuration to (optional)

        Returns:
            Path where the configuration was saved
        """
        if output_path is None:
            output_path = self.config_path

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Configuration saved to {output_path}")
        return output_path
```

---

### src/config/model_configs/__init__.py

```python
```

---

### src/config/model_configs/models.yaml

```yaml
```

---

### src/evaluation/__init__.py

```python
```

---

### src/evaluation/metrics.py

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import json
from pathlib import Path
from tqdm.auto import tqdm


def calculate_metrics(y_true, y_pred, y_pred_prob=None, class_names=None):
    """
    Calculate evaluation metrics for classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities (optional)
        class_names: List of class names (optional)

    Returns:
        Dictionary with calculated metrics
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred

    # Calculate basic metrics
    metrics = {
        "accuracy": accuracy_score(y_true_indices, y_pred_indices),
        "precision_macro": precision_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
    }

    # Calculate AUC if probabilities are provided
    if y_pred_prob is not None:
        # For multi-class classification
        if y_pred_prob.shape[1] > 2:
            try:
                # Convert y_true to one-hot encoding if it's not already
                if y_true.ndim == 1 or y_true.shape[1] == 1:
                    n_classes = y_pred_prob.shape[1]
                    y_true_onehot = np.zeros((len(y_true_indices), n_classes))
                    y_true_onehot[np.arange(len(y_true_indices)), y_true_indices] = 1
                else:
                    y_true_onehot = y_true

                metrics["roc_auc_macro"] = roc_auc_score(
                    y_true_onehot, y_pred_prob, average="macro", multi_class="ovr"
                )
                metrics["roc_auc_weighted"] = roc_auc_score(
                    y_true_onehot, y_pred_prob, average="weighted", multi_class="ovr"
                )
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")
        # For binary classification
        elif y_pred_prob.shape[1] == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true_indices, y_pred_prob[:, 1])
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")

    # Add per-class metrics if class names are provided
    if class_names is not None:
        # Get report as dictionary
        report = classification_report(
            y_true_indices,
            y_pred_indices,
            output_dict=True,
            target_names=(
                class_names
                if isinstance(class_names, list)
                else list(class_names.values())
            ),
        )

        # Add per-class metrics to the main metrics dictionary
        metrics["per_class"] = {}
        for class_name in report:
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                metrics["per_class"][class_name] = {
                    "precision": report[class_name]["precision"],
                    "recall": report[class_name]["recall"],
                    "f1-score": report[class_name]["f1-score"],
                    "support": report[class_name]["support"],
                }

    # Calculate confusion matrix but don't include in returned metrics
    # (it's not JSON serializable)
    cm = confusion_matrix(y_true_indices, y_pred_indices)

    return metrics


def evaluate_model(
    model, test_data, class_names=None, metrics_path=None, use_tqdm=True
):
    """
    Evaluate a model on test data with progress tracking

    Args:
        model: TensorFlow model to evaluate
        test_data: Test dataset
        class_names: List of class names (optional)
        metrics_path: Path to save metrics (optional)
        use_tqdm: Whether to use tqdm progress bar

    Returns:
        Dictionary with evaluation metrics
    """
    # Create predictions with progress bar
    print("Generating predictions...")

    if use_tqdm:
        # Get number of batches
        n_batches = len(test_data)

        # Initialize lists for predictions and true labels
        all_y_pred = []
        all_y_true = []

        # Use tqdm for progress tracking
        for batch_idx, (x, y) in enumerate(
            tqdm(test_data, desc="Predicting", total=n_batches)
        ):
            # Get predictions for this batch
            y_pred = model.predict(x, verbose=0)
            all_y_pred.append(y_pred)
            all_y_true.append(y)

        # Concatenate all batches
        y_pred_prob = np.vstack(all_y_pred)
        y_true = np.vstack(all_y_true)
    else:
        # Standard evaluation without progress tracking
        y_pred_prob = model.predict(test_data)
        y_true = np.concatenate([y for x, y in test_data], axis=0)

    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_prob, class_names)

    # Add standard evaluation metrics
    if hasattr(model, "evaluate"):
        results = model.evaluate(test_data, verbose=1)
        for i, metric_name in enumerate(model.metrics_names):
            metrics[metric_name] = results[i]

    # Save metrics if path is provided
    if metrics_path:
        # Convert path to Path object if it's a string
        if isinstance(metrics_path, str):
            metrics_path = Path(metrics_path)

        # Create parent directories if they don't exist
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python native types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if key == "per_class":
                metrics_json[key] = {}
                for class_name, class_metrics in value.items():
                    metrics_json[key][class_name] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in class_metrics.items()
                    }
            else:
                metrics_json[key] = (
                    float(value)
                    if isinstance(value, (np.float32, np.float64))
                    else value
                )

        # Save to file
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=4)

    return metrics
```

---

### src/evaluation/visualization.py

```python
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from pathlib import Path


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics over epochs"""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
    elif "acc" in history.history:  # For compatibility with older TF versions
        plt.plot(history.history["acc"], label="Training Accuracy")
        if "val_acc" in history.history:
            plt.plot(history.history["val_acc"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true, y_pred, class_names=None, save_path=None, figsize=(10, 8), normalize=False
):
    """
    Plot confusion matrix

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(cm.shape[0])]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(cm.shape[0])]
    else:
        labels = class_names

    # Truncate long class names
    max_length = 20
    labels = [
        label[:max_length] + "..." if len(label) > max_length else label
        for label in labels
    ]

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Add counts to plot title
    plt.figtext(0.5, 0.01, f"Total samples: {len(y_true)}", ha="center")

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true, y_pred_prob, class_names=None, save_path=None, figsize=(10, 8)
):
    """
    Plot ROC curve for multi-class classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
        y_true_onehot = y_true
    else:
        y_true_indices = y_true
        # Convert to one-hot encoding
        n_classes = y_pred_prob.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Calculate ROC curve and ROC area for each class
    n_classes = y_pred_prob.shape[1]

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(n_classes)]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(n_classes)]
    else:
        labels = class_names

    plt.figure(figsize=figsize)

    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{labels[i]} (AUC = {roc_auc:.2f})")

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    y_true, y_pred_prob, class_names=None, save_path=None, figsize=(10, 8)
):
    """
    Plot precision-recall curve for multi-class classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
        y_true_onehot = y_true
    else:
        y_true_indices = y_true
        # Convert to one-hot encoding
        n_classes = y_pred_prob.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Calculate Precision-Recall curve for each class
    n_classes = y_pred_prob.shape[1]

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(n_classes)]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(n_classes)]
    else:
        labels = class_names

    plt.figure(figsize=figsize)

    # Plot Precision-Recall curve for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, i], y_pred_prob[:, i]
        )
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{labels[i]} (AUC = {pr_auc:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_class_distribution(y_true, class_names=None, save_path=None, figsize=(12, 6)):
    """
    Plot class distribution

    Args:
        y_true: True labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Count occurrences of each class
    unique_classes, counts = np.unique(y_true, return_counts=True)

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in unique_classes]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in unique_classes]
    else:
        labels = [class_names[i] for i in unique_classes]

    # Sort by frequency
    idx = np.argsort(counts)[::-1]
    counts = counts[idx]
    labels = [labels[i] for i in idx]

    # Plot
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(counts)), counts, align="center")
    plt.xticks(range(len(counts)), labels, rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")

    # Add counts on top of bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 5,
            str(counts[i]),
            ha="center",
        )

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_misclassified_examples(
    x_test,
    y_true,
    y_pred,
    class_names=None,
    num_examples=9,
    save_path=None,
    figsize=(15, 15),
):
    """
    Plot misclassified examples

    Args:
        x_test: Test images
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        num_examples: Number of examples to plot
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Find misclassified examples
    misclassified = np.where(y_true != y_pred)[0]

    if len(misclassified) == 0:
        print("No misclassified examples found!")
        return

    # Process class names
    if class_names is None:
        # Use indices as class names
        get_class_name = lambda idx: str(idx)
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, use it directly
        get_class_name = lambda idx: class_names[idx]
    else:
        # If class_names is a list, use indices
        get_class_name = lambda idx: class_names[idx]

    # Select random misclassified examples
    indices = np.random.choice(
        misclassified, size=min(num_examples, len(misclassified)), replace=False
    )

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(len(indices))))

    # Plot
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        # Get the image
        img = x_test[idx]

        # For greyscale images
        if img.shape[-1] == 1:
            img = img.reshape(img.shape[:-1])

        # Normalize image if needed
        if img.max() > 1.0:
            img = img / 255.0

        # Plot the image
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {get_class_name(y_true[idx])}\nPred: {get_class_name(y_pred[idx])}"
        )
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
```

---

### src/model_registry/__init__.py

```python
"""
Model Registry for tracking and managing trained models.

This module provides functionality for:
- Registering and tracking trained models
- Managing model versions and runs
- Comparing model performance
- Generating reports and visualizations
"""

from model_registry.registry_manager import ModelRegistryManager

__all__ = ["ModelRegistryManager"]
```

---

### src/model_registry/registry_manager.py

```python
import os
import json
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from config.config import get_paths


class ModelRegistryManager:
    """
    Manager for the model registry that keeps track of all trained models
    in the trials folder, providing methods to register, retrieve, and compare models.
    """

    def __init__(self):
        """Initialize the model registry manager"""
        self.paths = get_paths()
        self.registry_file = self.paths.trials_dir / "registry.json"
        self._registry = self._load_registry()

    def _load_registry(self):
        """Load the registry from the JSON file or create a new one"""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        else:
            # Create default registry structure
            registry = {
                "models": {},
                "metadata": {
                    "version": "1.0.0",
                    "last_updated": datetime.now().isoformat(),
                    "total_models": 0,
                    "total_runs": 0,
                },
            }
            self._save_registry(registry)
            return registry

    def _save_registry(self, registry=None):
        """Save the registry to the JSON file"""
        if registry is None:
            registry = self._registry

        # Update metadata
        registry["metadata"]["last_updated"] = datetime.now().isoformat()
        registry["metadata"]["total_models"] = len(registry["models"])
        registry["metadata"]["total_runs"] = sum(
            len(model_info["runs"]) for model_info in registry["models"].values()
        )

        # Ensure the directory exists
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def scan_trials(self, rescan=False):
        """
        Scan the trials directory to discover models and runs that aren't
        in the registry yet.

        Args:
            rescan: If True, rescan all model directories even if they're in the registry

        Returns:
            Number of new runs added to the registry
        """
        trials_dir = self.paths.trials_dir
        if not trials_dir.exists():
            print(f"Trials directory {trials_dir} doesn't exist. Creating...")
            trials_dir.mkdir(parents=True, exist_ok=True)
            return 0

        # Get all model directories
        model_dirs = [
            d for d in trials_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

        # Initialize counters
        new_models = 0
        new_runs = 0

        # Iterate through model directories
        for model_dir in tqdm(model_dirs, desc="Scanning models"):
            model_name = model_dir.name

            # Skip if already in registry and not rescanning
            if not rescan and model_name in self._registry["models"]:
                continue

            # Add model to registry if needed
            if model_name not in self._registry["models"]:
                self._registry["models"][model_name] = {
                    "name": model_name,
                    "runs": {},
                    "best_run": None,
                    "last_run": None,
                    "total_runs": 0,
                }
                new_models += 1

            # Get all run directories
            run_dirs = [
                d
                for d in model_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ]

            # Iterate through run directories
            for run_dir in run_dirs:
                run_id = run_dir.name

                # Skip if already in registry and not rescanning
                if (
                    not rescan
                    and run_id in self._registry["models"][model_name]["runs"]
                ):
                    continue

                # Extract run information
                run_info = self._extract_run_info(model_name, run_id, run_dir)

                # Add to registry
                self._registry["models"][model_name]["runs"][run_id] = run_info
                self._registry["models"][model_name]["total_runs"] += 1
                new_runs += 1

                # Update last run
                self._registry["models"][model_name]["last_run"] = run_id

                # Update best run if needed
                if self._registry["models"][model_name]["best_run"] is None:
                    self._registry["models"][model_name]["best_run"] = run_id
                else:
                    best_run_id = self._registry["models"][model_name]["best_run"]
                    best_run = self._registry["models"][model_name]["runs"][best_run_id]
                    current_accuracy = best_run.get("metrics", {}).get(
                        "test_accuracy", 0
                    )
                    new_accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)

                    if new_accuracy > current_accuracy:
                        self._registry["models"][model_name]["best_run"] = run_id

        # Save registry if any changes were made
        if new_models > 0 or new_runs > 0:
            self._save_registry()
            print(f"Added {new_models} new models and {new_runs} new runs to registry")
        else:
            print("No new models or runs found")

        return new_runs

    def _extract_run_info(self, model_name, run_id, run_dir):
        """Extract information about a model run"""
        run_info = {
            "id": run_id,
            "path": str(run_dir),
            "timestamp": None,
            "metrics": {},
            "model_path": None,
            "has_checkpoints": False,
            "has_tensorboard": False,
            "status": "unknown",
        }

        # Extract timestamp from run_id
        try:
            timestamp_part = run_id.split("_")[1:3]
            run_info["timestamp"] = "_".join(timestamp_part)
        except:
            pass

        # Check for model file
        model_file = run_dir / f"{model_name}_final.h5"
        if model_file.exists():
            run_info["model_path"] = str(model_file)
            run_info["status"] = "completed"

        # Check for metrics file
        metrics_file = run_dir / "final_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    run_info["metrics"] = json.load(f)
            except:
                pass

        # Check for evaluation metrics
        eval_metrics_file = run_dir / "evaluation" / "metrics.json"
        if eval_metrics_file.exists():
            try:
                with open(eval_metrics_file, "r") as f:
                    eval_metrics = json.load(f)
                    # Add evaluation metrics with "eval_" prefix to avoid conflicts
                    for key, value in eval_metrics.items():
                        if key not in run_info["metrics"]:
                            # Only add if not already present
                            run_info["metrics"][f"eval_{key}"] = value
            except:
                pass

        # Check for checkpoints
        checkpoint_dir = run_dir / "training" / "checkpoints"
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            run_info["has_checkpoints"] = True

        # Check for tensorboard logs
        tensorboard_dir = run_dir / "training" / "tensorboard"
        if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
            run_info["has_tensorboard"] = True

        return run_info

    def register_model(self, model, model_name, metrics, history, run_dir):
        """
        Register a trained model in the registry

        Args:
            model: The trained TensorFlow model
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            history: Training history object
            run_dir: Directory where the model is saved

        Returns:
            Run ID of the registered model
        """
        # Convert run_dir to Path if it's a string
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        # Get run_id from directory name
        run_id = run_dir.name

        # Add model to registry if needed
        if model_name not in self._registry["models"]:
            self._registry["models"][model_name] = {
                "name": model_name,
                "runs": {},
                "best_run": None,
                "last_run": None,
                "total_runs": 0,
            }

        # Save the model if it hasn't been saved already
        model_path = run_dir / f"{model_name}_final.h5"
        if not model_path.exists():
            model.save(model_path)
            print(f"Model saved to {model_path}")

        # Save metrics to file if not already saved
        metrics_path = run_dir / "final_metrics.json"
        if not metrics_path.exists():
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        # Extract run information
        run_info = self._extract_run_info(model_name, run_id, run_dir)

        # Add to registry
        self._registry["models"][model_name]["runs"][run_id] = run_info
        self._registry["models"][model_name]["total_runs"] += 1
        self._registry["models"][model_name]["last_run"] = run_id

        # Update best run if needed
        if self._registry["models"][model_name]["best_run"] is None:
            self._registry["models"][model_name]["best_run"] = run_id
        else:
            best_run_id = self._registry["models"][model_name]["best_run"]
            best_run = self._registry["models"][model_name]["runs"][best_run_id]
            current_accuracy = best_run.get("metrics", {}).get("test_accuracy", 0)
            new_accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)

            if new_accuracy > current_accuracy:
                self._registry["models"][model_name]["best_run"] = run_id

        # Save registry
        self._save_registry()

        return run_id

    def get_model(self, model_name, run_id=None, best=False):
        """
        Get a model from the registry

        Args:
            model_name: Name of the model
            run_id: ID of the run to retrieve. If None, uses the latest run.
            best: If True, retrieves the best run instead of the latest

        Returns:
            Loaded TensorFlow model, or None if not found
        """
        # Check if model exists in registry
        if model_name not in self._registry["models"]:
            print(f"Model {model_name} not found in registry")
            return None

        # Determine run_id to retrieve
        if run_id is None:
            if best:
                run_id = self._registry["models"][model_name]["best_run"]
                if run_id is None:
                    print(f"No best run found for model {model_name}")
                    return None
            else:
                run_id = self._registry["models"][model_name]["last_run"]
                if run_id is None:
                    print(f"No runs found for model {model_name}")
                    return None
        elif run_id not in self._registry["models"][model_name]["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return None

        # Get model path
        run_info = self._registry["models"][model_name]["runs"][run_id]
        model_path = run_info.get("model_path")

        if model_path is None or not os.path.exists(model_path):
            print(f"Model file not found for {model_name} run {run_id}")
            return None

        # Load and return the model
        try:
            print(f"Loading model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model {model_name} run {run_id}: {e}")
            return None

    def get_run_info(self, model_name, run_id=None, best=False):
        """
        Get information about a specific run

        Args:
            model_name: Name of the model
            run_id: ID of the run to retrieve. If None, uses the latest run.
            best: If True, retrieves the best run instead of the latest

        Returns:
            Dictionary with run information, or None if not found
        """
        # Check if model exists in registry
        if model_name not in self._registry["models"]:
            print(f"Model {model_name} not found in registry")
            return None

        # Determine run_id to retrieve
        if run_id is None:
            if best:
                run_id = self._registry["models"][model_name]["best_run"]
                if run_id is None:
                    print(f"No best run found for model {model_name}")
                    return None
            else:
                run_id = self._registry["models"][model_name]["last_run"]
                if run_id is None:
                    print(f"No runs found for model {model_name}")
                    return None
        elif run_id not in self._registry["models"][model_name]["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return None

        # Return run information
        return self._registry["models"][model_name]["runs"][run_id]

    def list_models(self):
        """
        List all models in the registry

        Returns:
            List of model names
        """
        return list(self._registry["models"].keys())

    def list_runs(self, model_name):
        """
        List all runs for a specific model

        Args:
            model_name: Name of the model

        Returns:
            List of run IDs, or empty list if model not found
        """
        if model_name not in self._registry["models"]:
            print(f"Model {model_name} not found in registry")
            return []

        return list(self._registry["models"][model_name]["runs"].keys())

    def get_best_models(self, top_n=5, metric="test_accuracy"):
        """
        Get the best performing models according to a specific metric

        Args:
            top_n: Number of top models to return
            metric: Metric to use for ranking

        Returns:
            List of dictionaries with model information
        """
        # Collect best run for each model
        best_models = []

        for model_name, model_info in self._registry["models"].items():
            best_run_id = model_info["best_run"]
            if best_run_id is None:
                continue

            best_run = model_info["runs"][best_run_id]
            metric_value = best_run.get("metrics", {}).get(metric, 0)

            best_models.append(
                {
                    "name": model_name,
                    "run_id": best_run_id,
                    "metric": metric,
                    "value": metric_value,
                    "path": best_run.get("path"),
                    "timestamp": best_run.get("timestamp"),
                }
            )

        # Sort by metric value (highest first)
        best_models.sort(key=lambda x: x["value"], reverse=True)

        # Return top N
        return best_models[:top_n]

    def compare_models(
        self, model_names=None, metrics=None, plot=True, output_dir=None
    ):
        """
        Compare multiple models based on specified metrics

        Args:
            model_names: List of model names to compare. If None, uses all models.
            metrics: List of metrics to compare. If None, uses a default set.
            plot: Whether to generate comparison plots
            output_dir: Directory to save plots. If None, uses trials/comparisons.

        Returns:
            DataFrame with comparison results
        """
        # Set default metrics if none provided
        if metrics is None:
            metrics = ["test_accuracy", "test_loss", "training_time"]

        # Use all models if none specified
        if model_names is None:
            model_names = self.list_models()

        # Prepare data for comparison
        comparison_data = []

        for model_name in model_names:
            if model_name not in self._registry["models"]:
                print(f"Model {model_name} not found in registry")
                continue

            # Get best run
            best_run_id = self._registry["models"][model_name]["best_run"]
            if best_run_id is None:
                print(f"No best run found for model {model_name}")
                continue

            best_run = self._registry["models"][model_name]["runs"][best_run_id]

            # Collect metrics
            model_data = {
                "Model": model_name,
                "Run ID": best_run_id,
            }

            # Add each requested metric
            for metric in metrics:
                model_data[metric] = best_run.get("metrics", {}).get(metric, None)

            comparison_data.append(model_data)

        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Generate plots if requested
        if plot and len(comparison_data) > 0:
            if output_dir is None:
                output_dir = self.paths.trials_dir / "comparisons"

            # Ensure directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Plot each metric
            for metric in metrics:
                if metric in comparison_df.columns:
                    plt.figure(figsize=(12, 6))

                    # Sort by metric value
                    sorted_df = comparison_df.sort_values(by=metric, ascending=False)

                    # Create bar plot
                    ax = sns.barplot(x="Model", y=metric, data=sorted_df)

                    # Add values on top of bars
                    for i, v in enumerate(sorted_df[metric]):
                        if v is not None:
                            ax.text(i, v, f"{v:.4f}", ha="center")

                    plt.title(f"Model Comparison - {metric}")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()

                    # Save plot
                    plt.savefig(Path(output_dir) / f"comparison_{metric}.png")
                    plt.close()

            # Create a combined metrics plot
            plt.figure(figsize=(14, 8))

            # Number of metrics to plot
            valid_metrics = [m for m in metrics if m in comparison_df.columns]
            n_metrics = len(valid_metrics)

            if n_metrics > 0:
                # Normalize metrics for combined visualization
                normalized_df = comparison_df.copy()

                for metric in valid_metrics:
                    if metric in normalized_df.columns:
                        values = normalized_df[metric].dropna()
                        if len(values) > 0:
                            min_val = values.min()
                            max_val = values.max()
                            if max_val > min_val:
                                normalized_df[f"{metric}_norm"] = (
                                    normalized_df[metric] - min_val
                                ) / (max_val - min_val)
                            else:
                                normalized_df[f"{metric}_norm"] = 0

                # Plot normalized metrics
                plt.subplot(2, 1, 1)

                for i, metric in enumerate(valid_metrics):
                    norm_metric = f"{metric}_norm"
                    if norm_metric in normalized_df.columns:
                        plt.plot(
                            normalized_df["Model"],
                            normalized_df[norm_metric],
                            marker="o",
                            label=metric,
                        )

                plt.title("Normalized Metrics Comparison")
                plt.xticks(rotation=45, ha="right")
                plt.ylim(0, 1.1)
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Plot actual metrics in subplots
                for i, metric in enumerate(valid_metrics):
                    plt.subplot(2, n_metrics, n_metrics + i + 1)
                    if metric in comparison_df.columns:
                        sorted_df = comparison_df.sort_values(
                            by=metric, ascending=False
                        )
                        ax = sns.barplot(x="Model", y=metric, data=sorted_df)
                        plt.title(metric)
                        plt.xticks(rotation=45, ha="right")
                        plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(Path(output_dir) / "comparison_combined.png")
                plt.close()

        return comparison_df

    def delete_run(self, model_name, run_id, delete_files=False):
        """
        Delete a run from the registry

        Args:
            model_name: Name of the model
            run_id: ID of the run to delete
            delete_files: Whether to delete the run's files from disk

        Returns:
            True if successful, False otherwise
        """
        # Check if model exists in registry
        if model_name not in self._registry["models"]:
            print(f"Model {model_name} not found in registry")
            return False

        # Check if run exists
        if run_id not in self._registry["models"][model_name]["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return False

        # Get run information
        run_info = self._registry["models"][model_name]["runs"][run_id]
        run_path = run_info.get("path")

        # Delete files if requested
        if delete_files and run_path and os.path.exists(run_path):
            try:
                shutil.rmtree(run_path)
                print(f"Deleted run files at {run_path}")
            except Exception as e:
                print(f"Error deleting run files: {e}")
                return False

        # Remove run from registry
        del self._registry["models"][model_name]["runs"][run_id]
        self._registry["models"][model_name]["total_runs"] -= 1

        # Update best and last run
        if self._registry["models"][model_name]["best_run"] == run_id:
            # Find new best run
            best_run_id = None
            best_accuracy = -1

            for rid, rinfo in self._registry["models"][model_name]["runs"].items():
                accuracy = rinfo.get("metrics", {}).get("test_accuracy", 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_run_id = rid

            self._registry["models"][model_name]["best_run"] = best_run_id

        if self._registry["models"][model_name]["last_run"] == run_id:
            # Find new last run (most recent timestamp)
            last_run_id = None
            last_timestamp = ""

            for rid, rinfo in self._registry["models"][model_name]["runs"].items():
                timestamp = rinfo.get("timestamp", "")
                if timestamp > last_timestamp:
                    last_timestamp = timestamp
                    last_run_id = rid

            self._registry["models"][model_name]["last_run"] = last_run_id

        # Delete model if no runs left
        if len(self._registry["models"][model_name]["runs"]) == 0:
            del self._registry["models"][model_name]

        # Save registry
        self._save_registry()

        return True

    def export_registry(self, output_path=None):
        """
        Export the registry to a JSON file

        Args:
            output_path: Path to save the exported registry. If None, uses a timestamped name.

        Returns:
            Path to the exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.paths.trials_dir / f"registry_export_{timestamp}.json"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save registry
        with open(output_path, "w") as f:
            json.dump(self._registry, f, indent=2)

        print(f"Registry exported to {output_path}")
        return output_path

    def import_registry(self, input_path, merge=True):
        """
        Import a registry from a JSON file

        Args:
            input_path: Path to the registry file
            merge: Whether to merge with existing registry or replace it

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(input_path):
            print(f"Registry file {input_path} not found")
            return False

        try:
            with open(input_path, "r") as f:
                imported_registry = json.load(f)

            if merge:
                # Merge with existing registry
                for model_name, model_info in imported_registry["models"].items():
                    if model_name not in self._registry["models"]:
                        self._registry["models"][model_name] = model_info
                    else:
                        # Merge runs
                        for run_id, run_info in model_info["runs"].items():
                            if (
                                run_id
                                not in self._registry["models"][model_name]["runs"]
                            ):
                                self._registry["models"][model_name]["runs"][
                                    run_id
                                ] = run_info
                                self._registry["models"][model_name]["total_runs"] += 1
            else:
                # Replace existing registry
                self._registry = imported_registry

            # Save registry
            self._save_registry()

            print(f"Registry imported successfully from {input_path}")
            return True
        except Exception as e:
            print(f"Error importing registry: {e}")
            return False

    def generate_registry_report(self, output_path=None):
        """
        Generate an HTML report summarizing the registry contents

        Args:
            output_path: Path to save the report. If None, uses a default path.

        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = self.paths.trials_dir / "registry_report.html"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Registry Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .card {{
                    background: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                }}
                .model-card {{
                    margin-bottom: 30px;
                }}
                .best-run {{
                    background-color: #e8f4f8;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Registry Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="card">
                    <h2>Registry Summary</h2>
                    <table>
                        <tr>
                            <th>Total Models</th>
                            <td>{self._registry["metadata"]["total_models"]}</td>
                        </tr>
                        <tr>
                            <th>Total Runs</th>
                            <td>{self._registry["metadata"]["total_runs"]}</td>
                        </tr>
                        <tr>
                            <th>Last Updated</th>
                            <td>{self._registry["metadata"]["last_updated"]}</td>
                        </tr>
                    </table>
                </div>
        """

        # Get best models
        best_models = self.get_best_models(top_n=5)

        if best_models:
            html_content += """
                <div class="card">
                    <h2>Top Performing Models</h2>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Run ID</th>
                        </tr>
            """

            for i, model in enumerate(best_models):
                html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{model["name"]}</td>
                            <td>{model["value"]:.4f}</td>
                            <td>{model["run_id"]}</td>
                        </tr>
                """

            html_content += """
                    </table>
                </div>
            """

        # Add model details
        html_content += "<h2>Model Details</h2>"

        for model_name, model_info in self._registry["models"].items():
            best_run_id = model_info["best_run"]

            html_content += f"""
                <div class="card model-card">
                    <h3>{model_name}</h3>
                    <p><strong>Total Runs:</strong> {model_info["total_runs"]}</p>
                    
                    <h4>Runs:</h4>
                    <table>
                        <tr>
                            <th>Run ID</th>
                            <th>Timestamp</th>
                            <th>Accuracy</th>
                            <th>Loss</th>
                            <th>Status</th>
                        </tr>
                    """

            # Sort runs by timestamp (recent first)
            sorted_runs = sorted(
                model_info["runs"].items(),
                key=lambda x: x[1].get("timestamp", ""),
                reverse=True,
            )

            for run_id, run_info in sorted_runs:
                # Determine if this is the best run
                is_best = run_id == best_run_id
                row_class = "best-run" if is_best else ""

                # Get metrics
                accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)
                loss = run_info.get("metrics", {}).get("test_loss", 0)
                status = run_info.get("status", "unknown")

                html_content += f"""
                        <tr class="{row_class}">
                            <td>{run_id} {' (Best)' if is_best else ''}</td>
                            <td>{run_info.get("timestamp", "")}</td>
                            <td>{accuracy:.4f}</td>
                            <td>{loss:.4f}</td>
                            <td>{status}</td>
                        </tr>
                    """

            html_content += """
                    </table>
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        # Write the report
        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"Registry report generated at {output_path}")
        return output_path
```

---

### src/models/__init__.py

```python
# src/models/__init__.py
from src.models.model_factory import ModelFactory

__all__ = ["ModelFactory"]
```

---

### src/models/model_factory.py

```python
# models/model_factory.py
import tensorflow as tf

from tensorflow.keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    ResNet50,
    ResNet101,
    ResNet152,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    MobileNet,
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Small,
    InceptionV3,
    InceptionResNetV2,
    Xception,
    VGG16,
    VGG19,
)

from config.config_loader import ConfigLoader


class ModelFactory:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.models_dict = {
            # ConvNeXt models
            "ConvNeXtBase": self._create_convnext_base,
            "ConvNeXtLarge": self._create_convnext_large,
            "ConvNeXtSmall": self._create_convnext_small,
            "ConvNeXtTiny": self._create_convnext_tiny,
            "ConvNeXtXLarge": self._create_convnext_xlarge,
            # DenseNet models
            "DenseNet121": DenseNet121,
            "DenseNet169": DenseNet169,
            "DenseNet201": DenseNet201,
            # EfficientNet models
            "EfficientNetB0": EfficientNetB0,
            "EfficientNetB1": EfficientNetB1,
            "EfficientNetB2": EfficientNetB2,
            "EfficientNetB3": EfficientNetB3,
            "EfficientNetB4": EfficientNetB4,
            "EfficientNetB5": EfficientNetB5,
            "EfficientNetB6": EfficientNetB6,
            "EfficientNetB7": EfficientNetB7,
            # ResNet models
            "ResNet50": ResNet50,
            "ResNet101": ResNet101,
            "ResNet152": ResNet152,
            "ResNet50V2": tf.keras.applications.ResNet50V2,
            "ResNet101V2": tf.keras.applications.ResNet101V2,
            "ResNet152V2": tf.keras.applications.ResNet152V2,
            # MobileNet models
            "MobileNet": MobileNet,
            "MobileNetV2": MobileNetV2,
            "MobileNetV3Large": MobileNetV3Large,
            "MobileNetV3Small": MobileNetV3Small,
            # Others
            "InceptionV3": InceptionV3,
            "InceptionResNetV2": InceptionResNetV2,
            "Xception": Xception,
            "VGG16": VGG16,
            "VGG19": VGG19,
        }

    def get_model(self, model_name, num_classes, input_shape=None):
        """
        Create a model with the specified name and configuration
        """
        if model_name not in self.models_dict:
            raise ValueError(
                f"Model {model_name} not supported. Available models: {', '.join(self.models_dict.keys())}"
            )

        # Load model-specific configuration
        try:
            model_config = self.config_loader.get_model_config(model_name)
            config = model_config.get(model_name, {})
        except Exception as e:
            print(
                f"Warning: Could not load config for {model_name}: {e}. Using defaults."
            )
            config = {}

        # Use provided input shape or default from config
        if input_shape is None:
            input_shape = config.get("input_shape", (224, 224, 3))

        # Get base model constructor
        model_constructor = self.models_dict[model_name]

        print(f"Creating {model_name} model...")
        # Create the base model
        if callable(model_constructor):
            try:
                base_model = model_constructor(
                    include_top=config.get("include_top", False),
                    weights=config.get("weights", "imagenet"),
                    input_shape=input_shape,
                    pooling=config.get("pooling", "avg"),
                )
                print(f"Base model created successfully")
            except Exception as e:
                print(f"Error creating base model: {e}")
                raise
        else:
            # If it's a method, call it
            base_model = model_constructor()

        # Freeze layers if fine-tuning is enabled
        if config.get("fine_tuning", {}).get("enabled", False):
            freeze_layers = config["fine_tuning"].get("freeze_layers", 0)
            for layer in base_model.layers[:freeze_layers]:
                layer.trainable = False
            print(f"Froze {freeze_layers} layers for fine-tuning")

        # Build the full model with classification head
        x = base_model.output

        # Add dropout if specified
        if config.get("dropout_rate", 0) > 0:
            dropout_rate = config["dropout_rate"]
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            print(f"Added dropout with rate {dropout_rate}")

        # Add classification layer
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        # Create and return the model
        model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
        print(f"Final model created with {len(model.layers)} layers")

        return model

    # ConvNeXt models require special handling since they might not be
    # directly available in tf.keras.applications
    def _create_convnext_base(self, **kwargs):
        # Implementation depends on TensorFlow version
        try:
            return tf.keras.applications.convnext.ConvNeXtBase(**kwargs)
        except:
            # Fallback implementation if not available
            raise NotImplementedError(
                "ConvNeXtBase not available in this TensorFlow version"
            )

    def _create_convnext_large(self, **kwargs):
        try:
            return tf.keras.applications.convnext.ConvNeXtLarge(**kwargs)
        except:
            raise NotImplementedError(
                "ConvNeXtLarge not available in this TensorFlow version"
            )

    def _create_convnext_small(self, **kwargs):
        try:
            return tf.keras.applications.convnext.ConvNeXtSmall(**kwargs)
        except:
            raise NotImplementedError(
                "ConvNeXtSmall not available in this TensorFlow version"
            )

    def _create_convnext_tiny(self, **kwargs):
        try:
            return tf.keras.applications.convnext.ConvNeXtTiny(**kwargs)
        except:
            raise NotImplementedError(
                "ConvNeXtTiny not available in this TensorFlow version"
            )

    def _create_convnext_xlarge(self, **kwargs):
        try:
            return tf.keras.applications.convnext.ConvNeXtXLarge(**kwargs)
        except:
            raise NotImplementedError(
                "ConvNeXtXLarge not available in this TensorFlow version"
            )
```

---

### src/preprocessing/__init__.py

```python
```

---

### src/preprocessing/data_loader.py

```python
import tensorflow as tf
import os
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

from config.config import get_paths


class DataLoader:
    def __init__(self, config):
        """Initialize the data loader with configuration

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = get_paths()

        # Get training configuration
        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 32)
        self.validation_split = training_config.get("validation_split", 0.2)
        self.test_split = training_config.get("test_split", 0.1)

        # Get hardware configuration
        hardware_config = config.get("hardware", {})
        self.num_parallel_calls = hardware_config.get(
            "num_parallel_calls", tf.data.AUTOTUNE
        )
        self.prefetch_size = hardware_config.get(
            "prefetch_buffer_size", tf.data.AUTOTUNE
        )

        # Get data augmentation configuration
        self.augmentation_config = config.get("data_augmentation", {})

    def load_data(self, data_dir=None):
        """Load data from the specified directory.
        Assumes data is organized in a directory structure with each class in its own subdirectory.

        Args:
            data_dir: Path to the dataset directory. If None, uses the configured path.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset, class_names)
        """
        if data_dir is None:
            # Use configured paths
            data_path_config = self.config.get("paths", {}).get("data", {})
            if isinstance(data_path_config, dict):
                data_dir = data_path_config.get("processed", "data/processed")
            else:
                data_dir = "data/processed"

        # Ensure the path is absolute
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = self.paths.base_dir / data_dir

        print(f"Loading data from {data_dir}")

        # Set up image size based on configuration or default
        image_size = self.config.get("data", {}).get("image_size", (224, 224))
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        # Set up data augmentation parameters
        rotation_range = self.augmentation_config.get("rotation_range", 20)
        width_shift_range = self.augmentation_config.get("width_shift_range", 0.2)
        height_shift_range = self.augmentation_config.get("height_shift_range", 0.2)
        shear_range = self.augmentation_config.get("shear_range", 0.2)
        zoom_range = self.augmentation_config.get("zoom_range", 0.2)
        horizontal_flip = self.augmentation_config.get("horizontal_flip", True)
        vertical_flip = self.augmentation_config.get("vertical_flip", False)
        fill_mode = self.augmentation_config.get("fill_mode", "nearest")

        # Set up data generators with augmentation for training
        train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            fill_mode=fill_mode,
            validation_split=self.validation_split,
        )

        # Load training data with progress bar
        print("Loading training data...")
        train_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
        )

        # Load validation data with progress bar
        print("Loading validation data...")
        validation_generator = train_datagen.flow_from_directory(
            data_dir,
            target_size=image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
        )

        # Create a separate test set if specified
        test_generator = None
        if self.test_split > 0:
            test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rescale=1.0 / 255
            )

            # Check for test directory
            test_dir = self.paths.data_dir / "test"
            if test_dir.exists():
                print("Loading test data from dedicated test directory...")
                test_generator = test_datagen.flow_from_directory(
                    test_dir,
                    target_size=image_size,
                    batch_size=self.batch_size,
                    class_mode="categorical",
                    shuffle=False,
                )
            else:
                # Split validation data into validation and test
                print("Dedicated test directory not found.")
                print(
                    f"Using {self.test_split/(1-self.validation_split):.1%} of validation data as test set..."
                )

                # This is a simplified approach. In a real application, you would
                # want to create a proper split of the data.
                test_generator = validation_generator
        else:
            print("Test split not configured, skipping test data loading")

        # Get class names
        class_indices = train_generator.class_indices
        class_names = {v: k for k, v in class_indices.items()}

        # Print dataset statistics
        print(f"Dataset loaded successfully:")
        print(f"  - Training: {train_generator.samples} images")
        print(f"  - Validation: {validation_generator.samples} images")
        if test_generator:
            print(f"  - Test: {test_generator.samples} images")
        print(f"  - Classes: {len(class_names)} ({', '.join(class_names.values())})")
        print(f"  - Image size: {image_size}")
        print(f"  - Batch size: {self.batch_size}")

        return train_generator, validation_generator, test_generator, class_names

    def preprocess_function(self, image, label):
        """Apply preprocessing to a single example

        Args:
            image: Image tensor
            label: Label tensor

        Returns:
            Tuple of (processed_image, label)
        """
        # Get preprocessing configuration
        preprocessing_config = self.config.get("preprocessing", {})

        # Normalize if not already done
        if preprocessing_config.get("normalize", True):
            image = tf.cast(image, tf.float32) / 255.0

        # Apply additional preprocessing steps as configured
        if preprocessing_config.get("center_crop", False):
            # Center crop to target size
            target_size = preprocessing_config.get("target_size", (224, 224))
            image = tf.image.resize_with_crop_or_pad(
                image, target_size[0], target_size[1]
            )

        if preprocessing_config.get("standardize", False):
            # Standardize to mean 0, std 1
            image = tf.image.per_image_standardization(image)

        return image, label

    def apply_data_pipeline(self, dataset):
        """Apply preprocessing to a dataset with progress tracking

        Args:
            dataset: TensorFlow dataset

        Returns:
            Processed dataset
        """
        # Count items for progress bar
        total_items = sum(1 for _ in dataset.take(-1))

        with tqdm(total=total_items, desc="Preprocessing") as pbar:
            # Create a dataset that updates the progress bar
            def update_progress(*args):
                pbar.update(1)
                return args

            # Apply preprocessing with progress updates
            dataset = dataset.map(
                self.preprocess_function, num_parallel_calls=self.num_parallel_calls
            )
            dataset = dataset.map(update_progress)

            # Apply batching and prefetching
            dataset = dataset.batch(self.batch_size)
            dataset = dataset.prefetch(self.prefetch_size)

        return dataset
```

---

### src/scripts/__init__.py

```python
```

---

### src/scripts/evaluate.py

```python
#!/usr/bin/env python3
"""
Evaluate a trained model on a dataset
"""

import os
import argparse

import tensorflow as tf
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm


from src.config.config import get_paths
from src.config.config_loader import ConfigLoader
from src.preprocessing.data_loader import DataLoader
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution,
    plot_misclassified_examples,
)
from src.utils.report_generator import ReportGenerator


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.h5 file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the dataset directory for evaluation",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)",
    )
    parser.add_argument(
        "--visualize_misclassified",
        action="store_true",
        help="Generate visualizations of misclassified samples",
    )
    args = parser.parse_args()

    # Get project paths
    paths = get_paths()

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()

    # Override batch size if provided
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # If model path is inside trials directory, use its evaluation folder
        model_path = Path(args.model_path)
        model_dir = model_path.parent

        if "trials" in str(model_dir):
            output_dir = model_dir / "evaluation"
        else:
            # Default to a timestamp-based directory
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = paths.trials_dir / "evaluations" / f"eval_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation results will be saved to: {output_dir}")

    # Load model
    try:
        print(f"Loading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path)
        print(f"Model loaded successfully.")

        # Print model summary
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load data
    data_loader = DataLoader(config)
    _, _, test_data, class_names = data_loader.load_data(args.data_dir)

    if test_data is None:
        print("Error: No test data available for evaluation")
        return

    print(f"Evaluating model on {len(class_names)} classes")

    # Evaluate model
    print("Evaluating model...")
    metrics_path = output_dir / "metrics.json"
    metrics = evaluate_model(
        model,
        test_data,
        class_names=class_names,
        metrics_path=metrics_path,
        use_tqdm=True,
    )

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Loss: {metrics.get('loss', 0):.4f}")

    if "f1_macro" in metrics:
        print(f"  F1 Score (Macro): {metrics.get('f1_macro', 0):.4f}")

    if "precision_macro" in metrics:
        print(f"  Precision (Macro): {metrics.get('precision_macro', 0):.4f}")

    if "recall_macro" in metrics:
        print(f"  Recall (Macro): {metrics.get('recall_macro', 0):.4f}")

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions for visualization
    print("Generating predictions for visualization...")
    all_x = []
    all_y_true = []
    all_y_pred_prob = []

    # Use tqdm for progress tracking
    for batch_idx, (x, y) in enumerate(tqdm(test_data, desc="Predicting")):
        # Get predictions for this batch
        y_pred = model.predict(x, verbose=0)
        all_y_pred_prob.append(y_pred)
        all_y_true.append(y)

        # For misclassified visualization, save the images too
        if args.visualize_misclassified:
            all_x.append(x)

    # Concatenate all batches
    y_pred_prob = np.vstack(all_y_pred_prob)
    y_true = np.vstack(all_y_true)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # For misclassified visualization, concatenate the images
    if args.visualize_misclassified:
        x_test = np.vstack(all_x)

    # Plot confusion matrix
    print("Generating confusion matrix...")
    cm_path = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Plot normalized confusion matrix
    cm_norm_path = plots_dir / "confusion_matrix_normalized.png"
    plot_confusion_matrix(
        y_true, y_pred, class_names, save_path=cm_norm_path, normalize=True
    )
    print(f"Normalized confusion matrix saved to {cm_norm_path}")

    # Plot ROC curve
    print("Generating ROC curve...")
    roc_path = plots_dir / "roc_curve.png"
    plot_roc_curve(y_true, y_pred_prob, class_names, save_path=roc_path)
    print(f"ROC curve saved to {roc_path}")

    # Plot precision-recall curve
    print("Generating precision-recall curve...")
    pr_path = plots_dir / "precision_recall_curve.png"
    plot_precision_recall_curve(y_true, y_pred_prob, class_names, save_path=pr_path)
    print(f"Precision-recall curve saved to {pr_path}")

    # Plot class distribution
    print("Generating class distribution...")
    dist_path = plots_dir / "class_distribution.png"
    plot_class_distribution(y_true, class_names, save_path=dist_path)
    print(f"Class distribution saved to {dist_path}")

    # Plot misclassified examples if requested
    if args.visualize_misclassified:
        print("Generating misclassified examples visualization...")
        misclass_path = plots_dir / "misclassified_examples.png"
        plot_misclassified_examples(
            x_test,
            y_true,
            y_pred,
            class_names,
            num_examples=25,
            save_path=misclass_path,
        )
        print(f"Misclassified examples saved to {misclass_path}")

    # Generate HTML report
    print("Generating evaluation report...")
    report_generator = ReportGenerator(config)
    report_context = {
        "model_path": args.model_path,
        "metrics": metrics,
        "plots": {
            "confusion_matrix": str(cm_path),
            "confusion_matrix_normalized": str(cm_norm_path),
            "roc_curve": str(roc_path),
            "precision_recall_curve": str(pr_path),
            "class_distribution": str(dist_path),
        },
    }

    if args.visualize_misclassified:
        report_context["plots"]["misclassified_examples"] = str(misclass_path)

    report_path = output_dir / "evaluation_report.html"

    # Create a simple report template
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .metrics {{ margin: 20px 0; }}
            .metrics table {{ border-collapse: collapse; width: 100%; }}
            .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics th {{ background-color: #f2f2f2; }}
            .plots {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .plot {{ margin: 10px; text-align: center; }}
            .plot img {{ max-width: 800px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <p>Model: {os.path.basename(args.model_path)}</p>
        
        <h2>Metrics</h2>
        <div class="metrics">
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>{metrics.get('accuracy', 0):.4f}</td></tr>
                <tr><td>Loss</td><td>{metrics.get('loss', 0):.4f}</td></tr>
                <tr><td>F1 Score (Macro)</td><td>{metrics.get('f1_macro', 0):.4f}</td></tr>
                <tr><td>Precision (Macro)</td><td>{metrics.get('precision_macro', 0):.4f}</td></tr>
                <tr><td>Recall (Macro)</td><td>{metrics.get('recall_macro', 0):.4f}</td></tr>
            </table>
        </div>
        
        <h2>Visualizations</h2>
        <div class="plots">
            <div class="plot">
                <h3>Confusion Matrix</h3>
                <img src="{os.path.relpath(cm_path, output_dir)}" alt="Confusion Matrix">
            </div>
            
            <div class="plot">
                <h3>Normalized Confusion Matrix</h3>
                <img src="{os.path.relpath(cm_norm_path, output_dir)}" alt="Normalized Confusion Matrix">
            </div>
            
            <div class="plot">
                <h3>ROC Curve</h3>
                <img src="{os.path.relpath(roc_path, output_dir)}" alt="ROC Curve">
            </div>
            
            <div class="plot">
                <h3>Precision-Recall Curve</h3>
                <img src="{os.path.relpath(pr_path, output_dir)}" alt="Precision-Recall Curve">
            </div>
            
            <div class="plot">
                <h3>Class Distribution</h3>
                <img src="{os.path.relpath(dist_path, output_dir)}" alt="Class Distribution">
            </div>
            
            {f'<div class="plot"><h3>Misclassified Examples</h3><img src="{os.path.relpath(misclass_path, output_dir)}" alt="Misclassified Examples"></div>' if args.visualize_misclassified else ''}
        </div>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(report_html)

    print(f"Evaluation completed. Report generated at {report_path}")


if __name__ == "__main__":
    main()
```

---

### src/scripts/registry_cli.py

```python
#!/usr/bin/env python3
"""
Command line interface for the model registry.
This script allows you to manage the model registry from the command line.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.model_registry.registry_manager import ModelRegistryManager


def list_models(registry, args):
    """List all models in the registry"""
    models = registry.list_models()

    if not models:
        print("No models found in registry")
        return

    # Collect details for each model
    model_details = []
    for model_name in models:
        model_info = registry._registry["models"][model_name]
        best_run_id = model_info.get("best_run")

        if best_run_id:
            best_run = model_info["runs"][best_run_id]
            accuracy = best_run.get("metrics", {}).get("test_accuracy", 0)
            loss = best_run.get("metrics", {}).get("test_loss", 0)
        else:
            accuracy = 0
            loss = 0

        model_details.append(
            {
                "Model": model_name,
                "Total Runs": model_info.get("total_runs", 0),
                "Best Run": best_run_id,
                "Best Accuracy": f"{accuracy:.4f}",
                "Best Loss": f"{loss:.4f}",
            }
        )

    # Create DataFrame and display as table
    df = pd.DataFrame(model_details)
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


def list_runs(registry, args):
    """List all runs for a specific model"""
    model_name = args.model

    if model_name not in registry._registry["models"]:
        print(f"Model {model_name} not found in registry")
        return

    runs = registry.list_runs(model_name)
    model_info = registry._registry["models"][model_name]
    best_run_id = model_info.get("best_run")

    if not runs:
        print(f"No runs found for model {model_name}")
        return

    # Collect details for each run
    run_details = []
    for run_id in runs:
        run_info = model_info["runs"][run_id]
        accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)
        loss = run_info.get("metrics", {}).get("test_loss", 0)
        timestamp = run_info.get("timestamp", "")
        status = run_info.get("status", "unknown")
        is_best = run_id == best_run_id

        run_details.append(
            {
                "Run ID": run_id,
                "Timestamp": timestamp,
                "Accuracy": f"{accuracy:.4f}",
                "Loss": f"{loss:.4f}",
                "Status": status,
                "Best": "✓" if is_best else "",
            }
        )

    # Sort by timestamp (newest first)
    run_details.sort(key=lambda x: x["Timestamp"], reverse=True)

    # Create DataFrame and display as table
    df = pd.DataFrame(run_details)
    print(f"\nRuns for model: {model_name}")
    print(f"Total runs: {len(runs)}")
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


def show_details(registry, args):
    """Show detailed information about a model or run"""
    model_name = args.model
    run_id = args.run

    if model_name not in registry._registry["models"]:
        print(f"Model {model_name} not found in registry")
        return

    model_info = registry._registry["models"][model_name]

    if run_id:
        # Show run details
        if run_id not in model_info["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return

        run_info = model_info["runs"][run_id]
        print(f"\nDetails for {model_name} run {run_id}:")
        print(f"  Path: {run_info.get('path')}")
        print(f"  Timestamp: {run_info.get('timestamp')}")
        print(f"  Status: {run_info.get('status')}")
        print(f"  Model file: {run_info.get('model_path')}")
        print(f"  Has checkpoints: {run_info.get('has_checkpoints')}")
        print(f"  Has TensorBoard logs: {run_info.get('has_tensorboard')}")

        print("\nMetrics:")
        metrics = run_info.get("metrics", {})
        for key, value in metrics.items():
            # Format the value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
            print(f"  {key}: {formatted_value}")
    else:
        # Show model details
        print(f"\nDetails for model {model_name}:")
        print(f"  Total runs: {model_info.get('total_runs')}")
        print(f"  Best run: {model_info.get('best_run')}")
        print(f"  Last run: {model_info.get('last_run')}")

        if model_info.get("best_run"):
            best_run = model_info["runs"][model_info.get("best_run")]
            print("\nBest run metrics:")
            metrics = best_run.get("metrics", {})
            for key, value in metrics.items():
                if key.startswith(("test_", "val_")) and isinstance(
                    value, (int, float)
                ):
                    print(f"  {key}: {value:.6f}")


def scan_trials(registry, args):
    """Scan trials directory for new models and runs"""
    new_runs = registry.scan_trials(rescan=args.rescan)
    if new_runs > 0:
        print(f"Added {new_runs} new runs to registry")
    else:
        print("No new runs found")


def compare_models(registry, args):
    """Compare multiple models"""
    if args.models:
        model_names = args.models
    else:
        # Use top N models if no names provided
        top_models = registry.get_best_models(top_n=args.top)
        model_names = [model["name"] for model in top_models]

    if not model_names:
        print("No models to compare")
        return

    print(f"Comparing models: {', '.join(model_names)}")

    # Get metrics to compare
    if args.metrics:
        metrics = args.metrics
    else:
        metrics = ["test_accuracy", "test_loss", "training_time"]

    # Compare models
    comparison_df = registry.compare_models(
        model_names=model_names, metrics=metrics, plot=True, output_dir=args.output_dir
    )

    if comparison_df.empty:
        print("No data available for comparison")
        return

    # Display comparison table
    print("\nModel Comparison:")
    print(tabulate(comparison_df, headers="keys", tablefmt="pretty", showindex=False))

    # Print path to generated plots
    if args.output_dir:
        print(f"\nComparison plots saved to: {args.output_dir}")
    else:
        print(
            f"\nComparison plots saved to: {registry.paths.trials_dir / 'comparisons'}"
        )


def export_registry(registry, args):
    """Export the registry to a file"""
    output_path = args.output
    path = registry.export_registry(output_path)
    print(f"Registry exported to {path}")


def import_registry(registry, args):
    """Import a registry from a file"""
    input_path = args.input
    success = registry.import_registry(input_path, merge=not args.replace)
    if success:
        print(f"Registry imported from {input_path}")
    else:
        print(f"Failed to import registry from {input_path}")


def generate_report(registry, args):
    """Generate an HTML report of the registry"""
    output_path = args.output
    path = registry.generate_registry_report(output_path)
    print(f"Registry report generated at {path}")


def delete_run(registry, args):
    """Delete a run from the registry"""
    model_name = args.model
    run_id = args.run

    if not args.force:
        confirmation = input(
            f"Are you sure you want to delete run {run_id} of model {model_name}? (y/n): "
        )
        if confirmation.lower() != "y":
            print("Deletion cancelled")
            return

    success = registry.delete_run(model_name, run_id, delete_files=args.delete_files)
    if success:
        print(f"Run {run_id} of model {model_name} deleted from registry")
        if args.delete_files:
            print("Run files were also deleted from disk")
    else:
        print(f"Failed to delete run {run_id} of model {model_name}")


def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Model Registry CLI - Manage trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  registry_cli.py list                            # List all models
  registry_cli.py runs --model ResNet50           # List all runs for ResNet50
  registry_cli.py details --model ResNet50        # Show details for ResNet50
  registry_cli.py details --model ResNet50 --run run_20250304_123456_001  # Show run details
  registry_cli.py scan                            # Scan for new models and runs
  registry_cli.py compare --models ResNet50 MobileNetV2  # Compare models
  registry_cli.py report                          # Generate HTML report
        """,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List models command
    list_parser = subparsers.add_parser("list", help="List all models in the registry")

    # List runs command
    runs_parser = subparsers.add_parser(
        "runs", help="List all runs for a specific model"
    )
    runs_parser.add_argument("--model", required=True, help="Name of the model")

    # Show details command
    details_parser = subparsers.add_parser(
        "details", help="Show detailed information about a model or run"
    )
    details_parser.add_argument("--model", required=True, help="Name of the model")
    details_parser.add_argument("--run", help="ID of the run (optional)")

    # Scan trials command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan trials directory for new models and runs"
    )
    scan_parser.add_argument(
        "--rescan",
        action="store_true",
        help="Rescan all trials, even if already in registry",
    )

    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument(
        "--models", nargs="+", help="Names of models to compare"
    )
    compare_parser.add_argument("--metrics", nargs="+", help="Metrics to compare")
    compare_parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top models to compare if no models specified",
    )
    compare_parser.add_argument(
        "--output-dir", help="Directory to save comparison plots"
    )

    # Export registry command
    export_parser = subparsers.add_parser(
        "export", help="Export the registry to a file"
    )
    export_parser.add_argument("--output", help="Path to export the registry")

    # Import registry command
    import_parser = subparsers.add_parser(
        "import", help="Import a registry from a file"
    )
    import_parser.add_argument(
        "--input", required=True, help="Path to the registry file"
    )
    import_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing registry instead of merging",
    )

    # Generate report command
    report_parser = subparsers.add_parser(
        "report", help="Generate an HTML report of the registry"
    )
    report_parser.add_argument("--output", help="Path to save the report")

    # Delete run command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a run from the registry"
    )
    delete_parser.add_argument("--model", required=True, help="Name of the model")
    delete_parser.add_argument("--run", required=True, help="ID of the run")
    delete_parser.add_argument(
        "--delete-files", action="store_true", help="Also delete run files from disk"
    )
    delete_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return

    # Initialize registry manager
    registry = ModelRegistryManager()

    # Execute the appropriate command
    commands = {
        "list": list_models,
        "runs": list_runs,
        "details": show_details,
        "scan": scan_trials,
        "compare": compare_models,
        "export": export_registry,
        "import": import_registry,
        "report": generate_report,
        "delete": delete_run,
    }

    if args.command in commands:
        commands[args.command](registry, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

---

### src/scripts/train.py

```python
```

---

### src/training/__init__.py

```python
```

---

### src/training/trainer.py

```python
import os
import time
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

from src.utils.logger import Logger
from src.config.config import get_paths
from src.model_registry.registry_manager import ModelRegistryManager


class ProgressBarCallback(tf.keras.callbacks.Callback):
    """Custom callback for displaying training progress with tqdm"""

    def __init__(self, epochs, verbose=1):
        super(ProgressBarCallback, self).__init__()
        self.epochs = epochs
        self.verbose = verbose
        self.epoch_pbar = None
        self.batch_pbar = None

    def on_train_begin(self, logs=None):
        if self.verbose:
            self.epoch_pbar = tqdm(total=self.epochs, desc="Epochs", position=0)

    def on_train_end(self, logs=None):
        if self.verbose and self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose and hasattr(self.model, "train_step_count"):
            steps = getattr(self.model, "train_step_count")
            if self.batch_pbar is not None:
                self.batch_pbar.close()
            self.batch_pbar = tqdm(
                total=steps,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                position=1,
                leave=False,
            )

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            if self.epoch_pbar is not None:
                self.epoch_pbar.update(1)
                # Print metrics
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                self.epoch_pbar.set_postfix_str(metrics_str)

            if self.batch_pbar is not None:
                self.batch_pbar.close()
                self.batch_pbar = None

    def on_train_batch_end(self, batch, logs=None):
        if self.verbose and self.batch_pbar is not None:
            self.batch_pbar.update(1)
            if logs:
                # Show only loss and accuracy in batch progress
                metrics_to_show = {}
                if "loss" in logs:
                    metrics_to_show["loss"] = logs["loss"]
                if "accuracy" in logs:
                    metrics_to_show["acc"] = logs["accuracy"]

                if metrics_to_show:
                    metrics_str = ", ".join(
                        [f"{k}: {v:.4f}" for k, v in metrics_to_show.items()]
                    )
                    self.batch_pbar.set_postfix_str(metrics_str)


class Trainer:
    def __init__(self, config=None):
        """Initialize the trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = None
        self.paths = get_paths()

    def train(
        self, model, model_name, train_data, validation_data=None, test_data=None
    ):
        """Train a model and save results to the trials directory

        Args:
            model: TensorFlow model to train
            model_name: Name of the model
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            test_data: Test dataset (optional)

        Returns:
            Tuple of (model, history, metrics)
        """
        # Create run directory in trials folder
        run_dir = self.paths.get_model_trial_dir(model_name)

        # Initialize training logger
        self.train_logger = Logger(
            f"{model_name}",
            log_dir=run_dir,
            config=self.config.get("logging", {}),
            logger_type="training",
        )
        self.train_logger.log_config(self.config)
        self.train_logger.log_model_summary(model)

        # Initialize separate evaluation logger if configured
        if self.config.get("logging", {}).get("separate_loggers", True):
            self.eval_logger = Logger(
                f"{model_name}",
                log_dir=run_dir,
                config=self.config.get("logging", {}),
                logger_type="evaluation",
            )
        else:
            # Use the same logger for both if separate loggers not configured
            self.eval_logger = self.train_logger

        # Get training parameters from config
        training_config = self.config.get("training", {})
        batch_size = training_config.get("batch_size", 32)
        epochs = training_config.get("epochs", 50)
        learning_rate = training_config.get("learning_rate", 0.001)
        optimizer_name = training_config.get("optimizer", "adam").lower()
        loss = training_config.get("loss", "categorical_crossentropy")
        metrics = training_config.get("metrics", ["accuracy"])

        # Set up optimizer
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_name == "sgd":
            momentum = training_config.get("momentum", 0.9)
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=momentum
            )
        elif optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_name == "adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.train_logger.log_info(
            f"Model compiled with optimizer: {optimizer_name}, loss: {loss}, metrics: {metrics}"
        )

        # Set up callbacks
        callbacks = []

        # Get steps per epoch for progress bar
        steps_per_epoch = getattr(train_data, "samples", 0) // batch_size
        setattr(model, "train_step_count", steps_per_epoch)

        # Add progress bar callback
        progress_bar = ProgressBarCallback(epochs=epochs)
        callbacks.append(progress_bar)

        # Model checkpoint callback
        checkpoint_dir = Path(run_dir) / "training" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint-{epoch:02d}-{val_loss:.2f}.h5"

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            save_freq="epoch",
        )
        callbacks.append(checkpoint_callback)

        # TensorBoard callback
        tensorboard_dir = Path(run_dir) / "training" / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
        )
        callbacks.append(tensorboard_callback)

        # Early stopping if enabled
        early_stopping_config = training_config.get("early_stopping", {})
        if early_stopping_config.get("enabled", True):
            monitor = early_stopping_config.get("monitor", "val_loss")
            patience = early_stopping_config.get("patience", 10)
            restore_best_weights = early_stopping_config.get(
                "restore_best_weights", True
            )

            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=restore_best_weights,
                mode="min" if "loss" in monitor else "max",
            )
            callbacks.append(early_stopping_callback)
            self.train_logger.log_info(
                f"Early stopping enabled with patience {patience}, monitoring {monitor}"
            )

        # Learning rate scheduler if enabled
        lr_scheduler_config = training_config.get("lr_scheduler", {})
        if lr_scheduler_config.get("enabled", False):
            lr_scheduler_type = lr_scheduler_config.get("type", "reduce_on_plateau")

            if lr_scheduler_type == "reduce_on_plateau":
                reduce_factor = lr_scheduler_config.get("factor", 0.1)
                reduce_patience = lr_scheduler_config.get("patience", 5)
                reduce_min_lr = lr_scheduler_config.get("min_lr", 1e-6)

                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=reduce_factor,
                    patience=reduce_patience,
                    min_lr=reduce_min_lr,
                    verbose=1,
                )
                callbacks.append(lr_scheduler)
                self.train_logger.log_info(
                    f"LR scheduler enabled: ReduceLROnPlateau with factor {reduce_factor}"
                )

            elif lr_scheduler_type == "cosine_decay":
                decay_steps = lr_scheduler_config.get("decay_steps", epochs)
                alpha = lr_scheduler_config.get("alpha", 0.0)

                def cosine_decay_schedule(epoch, lr):
                    return learning_rate * (
                        alpha
                        + (1 - alpha) * np.cos(np.pi * epoch / decay_steps) / 2
                        + 0.5
                    )

                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                    cosine_decay_schedule
                )
                callbacks.append(lr_scheduler)
                self.train_logger.log_info(
                    f"LR scheduler enabled: Cosine decay over {decay_steps} epochs"
                )

        # Custom callback to log hardware metrics
        class HardwareMetricsCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger

            def on_epoch_end(self, epoch, logs=None):
                self.logger.log_hardware_metrics(step=epoch)

        callbacks.append(HardwareMetricsCallback(self.train_logger))

        # Train the model
        self.train_logger.log_info(
            f"Starting training for {model_name} with {epochs} epochs"
        )
        start_time = time.time()

        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,  # We're using our own progress bar
        )

        training_time = time.time() - start_time
        self.train_logger.log_info(f"Training completed in {training_time:.2f} seconds")

        # Evaluate on test set if provided
        test_metrics = {}
        if test_data is not None:
            self.train_logger.log_info("Evaluating on test data")
            self.eval_logger.log_info("Starting evaluation on test data")
            print("\nEvaluating on test data:")

            # Create a progress bar for evaluation
            test_steps = getattr(test_data, "samples", 0) // batch_size
            with tqdm(total=test_steps, desc="Evaluation") as pbar:
                # Custom callback to update progress bar during evaluation
                class EvalProgressCallback(tf.keras.callbacks.Callback):
                    def on_test_batch_end(self, batch, logs=None):
                        pbar.update(1)
                        if logs and "loss" in logs:
                            pbar.set_postfix(loss=f"{logs['loss']:.4f}")

                test_results = model.evaluate(
                    test_data, verbose=0, callbacks=[EvalProgressCallback()]
                )

            # Create metrics dictionary
            test_metrics = {}
            for i, metric_name in enumerate(model.metrics_names):
                test_metrics[f"test_{metric_name}"] = float(test_results[i])

            # Log test metrics to both loggers
            self.train_logger.log_metrics(test_metrics)
            self.eval_logger.log_metrics(test_metrics)
            self.eval_logger.log_info(
                f"Evaluation completed with accuracy: {test_metrics.get('test_accuracy', 'N/A')}"
            )
        # Combine metrics
        final_metrics = {
            "training_time": training_time,
            "run_dir": str(run_dir),
            **test_metrics,
        }

        # Save history metrics
        for key, values in history.history.items():
            # Save final (last epoch) value
            if values:
                final_metrics[f"final_{key}"] = float(values[-1])
                # Save best value for validation metrics
                if key.startswith("val_"):
                    metric_name = key[4:]  # Remove "val_" prefix
                    if metric_name in ["accuracy", "auc", "precision", "recall"]:
                        # For these metrics, higher is better
                        best_value = max(values)
                        best_epoch = values.index(best_value)
                    else:
                        # For loss and other metrics, lower is better
                        best_value = min(values)
                        best_epoch = values.index(best_value)

                    final_metrics[f"best_{key}"] = float(best_value)
                    final_metrics[f"best_{key}_epoch"] = best_epoch

        # Save final model
        model_path = Path(run_dir) / f"{model_name}_final.h5"
        model.save(str(model_path))
        final_metrics["model_path"] = str(model_path)

        # Save history to CSV
        history_df = pd.DataFrame(history.history)
        history_path = Path(run_dir) / "training" / "history.csv"
        history_df.to_csv(history_path, index=False)

        # Save metrics
        self.train_logger.save_final_metrics(final_metrics)

        # Save evaluation metrics separately if using a different logger
        if self.eval_logger != self.train_logger and test_metrics:
            # Add training time to evaluation metrics
            eval_metrics = {
                "training_time": training_time,
                "run_dir": str(run_dir),
                **test_metrics,
            }
            self.eval_logger.save_final_metrics(eval_metrics)

        # Generate confusion matrix if test data is available
        if test_data is not None and self.config.get("reporting", {}).get(
            "save_confusion_matrix", True
        ):
            try:
                # Get predictions
                self.eval_logger.log_info("Generating evaluation visualizations...")
                y_pred = model.predict(test_data, verbose=0)
                # Get true labels (assuming they're in the second element of the tuple)
                y_true = np.concatenate([y for _, y in test_data], axis=0)

                # Calculate confusion matrix
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_true, axis=1)

                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(y_true_classes, y_pred_classes)

                # Get class names if available
                class_info = getattr(test_data, "class_indices", None)
                if class_info:
                    class_names = {v: k for k, v in class_info.items()}
                else:
                    class_names = {i: f"Class {i}" for i in range(cm.shape[0])}

                # Log confusion matrix to evaluation logger
                self.eval_logger.log_confusion_matrix(
                    cm,
                    [class_names[i] for i in range(len(class_names))],
                    step=epochs - 1,
                )

                # Calculate additional metrics if requested
                if self.config.get("reporting", {}).get(
                    "save_roc_curves", True
                ) or self.config.get("reporting", {}).get(
                    "save_precision_recall", True
                ):
                    from evaluation.metrics import calculate_metrics

                    detailed_metrics = calculate_metrics(
                        y_true, y_pred_classes, y_pred, class_names
                    )

                    # Save detailed metrics
                    metrics_path = (
                        Path(run_dir) / "evaluation" / "detailed_metrics.json"
                    )
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)

                    import json

                    with open(metrics_path, "w") as f:
                        json.dump(detailed_metrics, f, indent=4)

                    # Generate visualization plots
                    from evaluation.visualization import (
                        plot_roc_curve,
                        plot_precision_recall_curve,
                        plot_confusion_matrix,
                    )

                    plots_dir = Path(run_dir) / "evaluation" / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    if self.config.get("reporting", {}).get(
                        "save_confusion_matrix", True
                    ):
                        cm_path = plots_dir / "confusion_matrix.png"
                        plot_confusion_matrix(
                            y_true, y_pred_classes, class_names, save_path=cm_path
                        )
                        self.eval_logger.log_info(
                            f"Confusion matrix saved to {cm_path}"
                        )

                    if self.config.get("reporting", {}).get("save_roc_curves", True):
                        roc_path = plots_dir / "roc_curve.png"
                        plot_roc_curve(y_true, y_pred, class_names, save_path=roc_path)
                        self.eval_logger.log_info(f"ROC curves saved to {roc_path}")

                    if self.config.get("reporting", {}).get(
                        "save_precision_recall", True
                    ):
                        pr_path = plots_dir / "precision_recall_curve.png"
                        plot_precision_recall_curve(
                            y_true, y_pred, class_names, save_path=pr_path
                        )
                        self.eval_logger.log_info(
                            f"Precision-recall curves saved to {pr_path}"
                        )

            except Exception as e:
                self.eval_logger.log_warning(
                    f"Error generating evaluation visualizations: {e}"
                )
                import traceback

                self.eval_logger.log_debug(traceback.format_exc())

        # Register model in the registry
        try:
            registry = ModelRegistryManager()
            registry.register_model(model, model_name, final_metrics, history, run_dir)
            self.train_logger.log_info(f"Model registered in registry")
        except Exception as e:
            self.train_logger.log_warning(f"Failed to register model in registry: {e}")

        return model, history, final_metrics
```

---

### src/utils/__init__.py

```python
```

---

### src/utils/logger.py

```python
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
```

---

### src/utils/report_generator.py

```python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from jinja2 import Template


from src.config.config import get_paths


class ReportGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        self.paths = get_paths()

    def generate_model_report(
        self, model_name, run_dir, metrics, history=None, class_names=None
    ):
        """Generate an HTML report for a model run"""
        # Convert run_dir to Path if it's a string
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        # Load metrics
        if isinstance(metrics, str) and os.path.exists(metrics):
            with open(metrics, "r") as f:
                metrics = json.load(f)

        # Load history if path provided
        if isinstance(history, str) and os.path.exists(history):
            history_df = pd.read_csv(history)
            history_dict = {col: history_df[col].tolist() for col in history_df.columns}
            history = type("obj", (object,), {"history": history_dict})

        # Create plots directory if needed
        plots_dir = run_dir / "training" / "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Generate plots if history is available
        if history is not None and hasattr(history, "history"):
            # Import here to avoid circular imports
            from src.evaluation.visualization import plot_training_history

            history_plot_path = plots_dir / "training_history.png"
            plot_training_history(history, save_path=history_plot_path)

            # Create additional plots if enabled in config
            if self.config.get("reporting", {}).get("generate_plots", True):
                # Generate learning rate plot if available
                if "lr" in history.history:
                    self._plot_learning_rate(history, plots_dir / "learning_rate.png")

                # Generate loss and metrics comparison plots
                metrics_to_plot = [
                    k
                    for k in history.history.keys()
                    if not k.startswith("lr") and not k.startswith("val_")
                ]

                for metric in metrics_to_plot:
                    val_metric = f"val_{metric}"
                    if val_metric in history.history:
                        self._plot_metric_comparison(
                            history,
                            metric,
                            val_metric,
                            plots_dir / f"{metric}_comparison.png",
                        )

        # Create report context
        context = {
            "model_name": model_name,
            "run_dir": str(run_dir),
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "has_history": history is not None and hasattr(history, "history"),
            "class_names": class_names,
            "project_name": self.config.get("project", {}).get(
                "name", "Plant Disease Detection"
            ),
            "project_version": self.config.get("project", {}).get("version", "1.0.0"),
        }

        # Generate HTML report using template
        html = self._render_report_template(context)

        # Save report to file
        report_path = run_dir / "report.html"
        with open(report_path, "w") as f:
            f.write(html)

        return report_path

    def _plot_learning_rate(self, history, save_path):
        """Plot learning rate over epochs"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["lr"])
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_metric_comparison(self, history, train_metric, val_metric, save_path):
        """Plot comparison between training and validation metrics"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history[train_metric], label=f"Training {train_metric}")
        plt.plot(history.history[val_metric], label=f"Validation {val_metric}")
        plt.title(f"{train_metric} Comparison")
        plt.xlabel("Epoch")
        plt.ylabel(train_metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _render_report_template(self, context):
        """Render HTML report using Jinja2 template"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ model_name }} Training Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    background-color: #f8f9fa;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .card {
                    background: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .header h1 {
                    color: white;
                    margin: 0;
                }
                .header p {
                    margin: 5px 0 0 0;
                    opacity: 0.8;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                }
                .metric-value {
                    font-weight: bold;
                }
                .good-metric {
                    color: #28a745;
                }
                .average-metric {
                    color: #fd7e14;
                }
                .poor-metric {
                    color: #dc3545;
                }
                .plot-container {
                    text-align: center;
                    margin: 20px 0;
                }
                .plot-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .plot-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .footer {
                    margin-top: 30px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    padding: 20px;
                    border-top: 1px solid #eee;
                }
                .summary {
                    font-size: 1.2em;
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f1f8ff;
                    border-left: 4px solid #4285f4;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ model_name }} Training Report</h1>
                    <p>{{ project_name }} v{{ project_version }} | Generated on {{ generation_time }}</p>
                </div>
                
                <div class="card">
                    <h2>Model Information</h2>
                    <table>
                        <tr>
                            <th>Model Name</th>
                            <td>{{ model_name }}</td>
                        </tr>
                        <tr>
                            <th>Run Directory</th>
                            <td>{{ run_dir }}</td>
                        </tr>
                        {% if "training_time" in metrics %}
                        <tr>
                            <th>Training Time</th>
                            <td>{{ "%.2f"|format(metrics["training_time"]) }} seconds ({{ "%.2f"|format(metrics["training_time"]/60) }} minutes)</td>
                        </tr>
                        {% endif %}
                        {% if class_names %}
                        <tr>
                            <th>Classes</th>
                            <td>{{ class_names|length }} classes
                                {% if class_names|length <= 20 %}
                                    <br><small>{{ class_names|join(', ') }}</small>
                                {% endif %}
                            </td>
                        </tr>
                        {% endif %}
                    </table>
                </div>
                
                {% if "test_accuracy" in metrics or "val_accuracy" in metrics %}
                <div class="summary">
                    Model Performance Summary: 
                    {% if "test_accuracy" in metrics %}
                        Test Accuracy: <span class="metric-value 
                            {% if metrics["test_accuracy"] > 0.9 %}good-metric
                            {% elif metrics["test_accuracy"] > 0.8 %}average-metric
                            {% else %}poor-metric{% endif %}">
                            {{ "%.2f"|format(metrics["test_accuracy"] * 100) }}%
                        </span>
                    {% elif "val_accuracy" in metrics %}
                        Validation Accuracy: <span class="metric-value 
                            {% if metrics["val_accuracy"] > 0.9 %}good-metric
                            {% elif metrics["val_accuracy"] > 0.8 %}average-metric
                            {% else %}poor-metric{% endif %}">
                            {{ "%.2f"|format(metrics["val_accuracy"] * 100) }}%
                        </span>
                    {% endif %}
                </div>
                {% endif %}
                
                <div class="card">
                    <h2>Performance Metrics</h2>
                    <table>
                        {% for key, value in metrics.items() %}
                        <tr>
                            <th>{{ key }}</th>
                            <td class="metric-value">
                                {% if value is number %}
                                    {{ "%.4f"|format(value) }}
                                    {% if "accuracy" in key or "precision" in key or "recall" in key or "f1" in key or "auc" in key %}
                                        ({{ "%.2f"|format(value * 100) }}%)
                                    {% endif %}
                                {% else %}
                                    {{ value }}
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                {% if has_history %}
                <div class="card">
                    <h2>Training Visualization</h2>
                    <div class="plot-container">
                        <img src="training/plots/training_history.png" alt="Training History">
                    </div>
                    
                    <h3>Additional Plots</h3>
                    <div class="plot-grid">
                        {% if metrics.get("loss") %}
                        <div class="plot-container">
                            <img src="training/plots/loss_comparison.png" alt="Loss Comparison">
                        </div>
                        {% endif %}
                        
                        {% if metrics.get("accuracy") %}
                        <div class="plot-container">
                            <img src="training/plots/accuracy_comparison.png" alt="Accuracy Comparison">
                        </div>
                        {% endif %}
                        
                        {% if metrics.get("lr") %}
                        <div class="plot-container">
                            <img src="training/plots/learning_rate.png" alt="Learning Rate">
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endif %}
                
                <div class="footer">
                    <p>{{ project_name }} | Deep Learning for Plant Disease Detection</p>
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(template_str)
        return template.render(**context)

    def generate_comparison_report(self, models_data, output_path=None):
        """Generate a comparison report for multiple models

        Args:
            models_data: List of dictionaries with model results
            output_path: Path to save the report
        """
        if output_path is None:
            output_path = self.paths.trials_dir / "model_comparison.html"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create comparison plots
        plots_dir = Path(os.path.dirname(output_path) / "comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Generate accuracy comparison plot
        accuracy_plot_path = Path(plots_dir / "accuracy_comparison.png")
        self._plot_model_comparison(
            models_data, "test_accuracy", "Test Accuracy", accuracy_plot_path
        )

        # Generate other comparison plots if data is available
        metrics_to_compare = [
            ("test_loss", "Test Loss"),
            ("precision_macro", "Precision (Macro)"),
            ("recall_macro", "Recall (Macro)"),
            ("f1_macro", "F1 Score (Macro)"),
            ("training_time", "Training Time (s)"),
        ]

        plot_paths = {"accuracy": accuracy_plot_path}

        for metric_key, metric_name in metrics_to_compare:
            if any(metric_key in model["metrics"] for model in models_data):
                plot_path = Path(plots_dir / f"{metric_key}_comparison.png")
                self._plot_model_comparison(
                    models_data, metric_key, metric_name, plot_path
                )
                plot_paths[metric_key] = plot_path

        # Create comparison context
        context = {
            "models_data": models_data,
            "plot_paths": plot_paths,
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_name": self.config.get("project", {}).get(
                "name", "Plant Disease Detection"
            ),
            "project_version": self.config.get("project", {}).get("version", "1.0.0"),
        }

        # Generate HTML report
        html = self._render_comparison_template(context)

        # Save report
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _plot_model_comparison(self, models_data, metric_key, metric_name, save_path):
        """Create a bar chart comparing models based on a metric"""
        # Extract data
        model_names = []
        metric_values = []

        for model in models_data:
            model_names.append(model["name"])
            if metric_key in model["metrics"]:
                metric_values.append(model["metrics"][metric_key])
            else:
                metric_values.append(0)

        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values)

        # Add value annotations
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            if (
                "accuracy" in metric_key
                or "precision" in metric_key
                or "recall" in metric_key
                or "f1" in metric_key
            ):
                text = f"{value:.2%}"
            else:
                text = f"{value:.2f}"
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                text,
                ha="center",
                va="bottom",
                rotation=0,
            )

        plt.title(f"Model Comparison - {metric_name}")
        plt.xlabel("Model")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path)
        plt.close()

    def _render_comparison_template(self, context):
        """Render HTML comparison report using Jinja2 template"""
        # Template implementation for comparison report
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                /* CSS styles from the model report template, plus comparison-specific styles */
                /* ... add styles as in the previous template ... */
                .comparison-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 30px;
                }
                .comparison-table th, .comparison-table td {
                    padding: 12px 15px;
                    text-align: center;
                    border: 1px solid #ddd;
                }
                .comparison-table th {
                    background-color: #f8f9fa;
                }
                .comparison-table tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .best-value {
                    font-weight: bold;
                    color: #28a745;
                }
                .second-best {
                    color: #17a2b8;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Model Comparison Report</h1>
                    <p>{{ project_name }} v{{ project_version }} | Generated on {{ generation_time }}</p>
                </div>
                
                <div class="card">
                    <h2>Models Compared</h2>
                    <p>Comparison of {{ models_data|length }} models</p>
                    
                    <div class="comparison-table-container">
                        <table class="comparison-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Test Accuracy</th>
                                    <th>Test Loss</th>
                                    <th>F1 Score</th>
                                    <th>Training Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for model in models_data %}
                                <tr>
                                    <td>{{ model.name }}</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("test_accuracy", 0) * 100) }}%</td>
                                    <td>{{ "%.4f"|format(model.metrics.get("test_loss", 0)) }}</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("f1_macro", 0) * 100) }}%</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("training_time", 0)) }}s</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Visual Comparison</h2>
                    
                    {% if plot_paths.accuracy %}
                    <div class="plot-container">
                        <h3>Accuracy Comparison</h3>
                        <img src="{{ plot_paths.accuracy }}" alt="Accuracy Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.test_loss %}
                    <div class="plot-container">
                        <h3>Loss Comparison</h3>
                        <img src="{{ plot_paths.test_loss }}" alt="Loss Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.f1_macro %}
                    <div class="plot-container">
                        <h3>F1 Score Comparison</h3>
                        <img src="{{ plot_paths.f1_macro }}" alt="F1 Score Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.training_time %}
                    <div class="plot-container">
                        <h3>Training Time Comparison</h3>
                        <img src="{{ plot_paths.training_time }}" alt="Training Time Comparison">
                    </div>
                    {% endif %}
                </div>
                
                <div class="summary">
                    <h2>Recommendation</h2>
                    {% set best_model = {'name': '', 'accuracy': 0} %}
                    {% for model in models_data %}
                        {% if model.metrics.get("test_accuracy", 0) > best_model.accuracy %}
                            {% set _ = best_model.update({'name': model.name, 'accuracy': model.metrics.get("test_accuracy", 0)}) %}
                        {% endif %}
                    {% endfor %}
                    
                    <p>Based on the comparison, <strong>{{ best_model.name }}</strong> performs best with an accuracy of {{ "%.2f"|format(best_model.accuracy * 100) }}%.</p>
                </div>
                
                <div class="footer">
                    <p>{{ project_name }} | Deep Learning for Plant Disease Detection</p>
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(template_str)
        return template.render(**context)
```

---

## Summary

Total files: 28
- Python files: 25
- YAML files: 3
