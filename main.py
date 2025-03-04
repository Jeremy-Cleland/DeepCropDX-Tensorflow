# main.py with enhancements

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
from src.preprocessing.data_loader_tf import TFDataLoader
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.utils.report_generator import ReportGenerator
from src.utils.logger import Logger
from src.model_registry.registry_manager import ModelRegistryManager
from src.utils.seed_utils import set_global_seeds


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
    resume=False,
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
        resume: Whether to resume training from checkpoints if available

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
            model, model_name, train_data, val_data, test_data, resume=resume
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
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--use_tf_data",
        action="store_true",
        help="Use the TF Data pipeline for loading data",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from checkpoints if available",
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
    if args.seed:
        config["seed"] = args.seed
        print(f"Overriding random seed: {args.seed}")
        batch_logger.log_info(f"Overriding random seed: {args.seed}")

    # Set up global random seeds
    seed = config.get("seed", 42)
    set_global_seeds(seed)
    batch_logger.log_info(f"Global random seeds set to {seed}")

    # Configure hardware
    configure_hardware(config)
    batch_logger.log_hardware_metrics(step=0)

    # Log system information
    batch_logger.log_info("Loading datasets...")

    # Load data (only once for all models)
    if args.use_tf_data:
        data_loader = TFDataLoader(config)
        batch_logger.log_info("Using TensorFlow Data API for dataset loading")
    else:
        from src.preprocessing.data_loader import DataLoader

        data_loader = DataLoader(config)
        batch_logger.log_info("Using Keras ImageDataGenerator for dataset loading")

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
            resume=args.resume,
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
        "seed": seed,
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


# import os
# import argparse
# import yaml
# import tensorflow as tf
# from pathlib import Path
# from tqdm.auto import tqdm
# import time
# from datetime import datetime

# from src.config.config import get_paths
# from src.config.config_loader import ConfigLoader
# from src.preprocessing.data_loader import DataLoader
# from src.models.model_factory import ModelFactory
# from src.training.trainer import Trainer
# from src.utils.report_generator import ReportGenerator
# from src.utils.logger import Logger
# from src.model_registry.registry_manager import ModelRegistryManager


# def configure_hardware(config):
#     """Configure TensorFlow for hardware acceleration

#     Args:
#         config: Configuration dictionary with hardware settings
#     """
#     hardware_config = config.get("hardware", {})

#     # Configure TensorFlow for Metal on Apple Silicon
#     if (
#         hardware_config.get("use_metal", True)
#         and hasattr(tf.test, "is_built_with_metal")
#         and tf.config.list_physical_devices("GPU")
#     ):
#         print("Configuring TensorFlow for Metal on Apple Silicon")

#         # Enable Metal
#         try:
#             tf.config.experimental.set_visible_devices(
#                 tf.config.list_physical_devices("GPU")[0], "GPU"
#             )

#             # Enable memory growth to prevent allocating all GPU memory at once
#             if hardware_config.get("memory_growth", True):
#                 for gpu in tf.config.list_physical_devices("GPU"):
#                     tf.config.experimental.set_memory_growth(gpu, True)

#             # Use mixed precision if enabled
#             if hardware_config.get("mixed_precision", True):
#                 tf.keras.mixed_precision.set_global_policy("mixed_float16")
#                 print("Mixed precision enabled (float16)")
#         except Exception as e:
#             print(f"Warning: Error configuring Metal: {e}")

#     # For CUDA GPUs
#     elif tf.config.list_physical_devices("GPU") and hardware_config.get(
#         "use_gpu", True
#     ):
#         print("Configuring TensorFlow for CUDA GPU")

#         try:
#             # Enable memory growth to prevent allocating all GPU memory at once
#             if hardware_config.get("memory_growth", True):
#                 for gpu in tf.config.list_physical_devices("GPU"):
#                     tf.config.experimental.set_memory_growth(gpu, True)

#             # Use mixed precision if enabled
#             if hardware_config.get("mixed_precision", True):
#                 tf.keras.mixed_precision.set_global_policy("mixed_float16")
#                 print("Mixed precision enabled (float16)")
#         except Exception as e:
#             print(f"Warning: Error configuring GPU: {e}")
#     else:
#         print("Using CPU for computation")

#     # Configure other TensorFlow options
#     tf.config.threading.set_inter_op_parallelism_threads(
#         hardware_config.get("inter_op_parallelism", 0)
#     )
#     tf.config.threading.set_intra_op_parallelism_threads(
#         hardware_config.get("intra_op_parallelism", 0)
#     )


# def train_model(
#     model_name,
#     config,
#     data_loader,
#     model_factory,
#     train_data,
#     val_data,
#     test_data,
#     class_names,
#     batch_logger=None,
# ):
#     """Train a single model and return results

#     Args:
#         model_name: Name of the model to train
#         config: Configuration dictionary
#         data_loader: DataLoader instance
#         model_factory: ModelFactory instance
#         train_data: Training dataset
#         val_data: Validation dataset
#         test_data: Test dataset
#         class_names: Dictionary of class names
#         batch_logger: Optional logger for batch training process

#     Returns:
#         Tuple of (success_flag, metrics_dict)
#     """
#     print(f"\n{'='*80}")
#     print(f"Training model: {model_name}")
#     print(f"{'='*80}")

#     if batch_logger:
#         batch_logger.log_info(f"Starting training for model: {model_name}")

#     try:
#         # Get model hyperparameters (combining defaults with model-specific)
#         config_loader = ConfigLoader()
#         hyperparams = config_loader.get_hyperparameters(model_name, config)

#         # Update config with these hyperparameters
#         training_config = config.get("training", {}).copy()
#         training_config.update(hyperparams)
#         config["training"] = training_config

#         print(f"Training configuration for {model_name}:")
#         for key, value in training_config.items():
#             print(f"  {key}: {value}")

#         if batch_logger:
#             batch_logger.log_info(
#                 f"Hyperparameters for {model_name}: {training_config}"
#             )

#         # Create model
#         model = model_factory.get_model(model_name, num_classes=len(class_names))

#         # Train model
#         trainer = Trainer(config)
#         model, history, metrics = trainer.train(
#             model, model_name, train_data, val_data, test_data
#         )

#         # Generate report
#         if config.get("reporting", {}).get("generate_html_report", True):
#             report_generator = ReportGenerator(config)
#             run_dir = metrics.get("run_dir", "")
#             report_path = report_generator.generate_model_report(
#                 model_name, run_dir, metrics, history, class_names
#             )
#             print(f"Report generated at {report_path}")

#             if batch_logger:
#                 batch_logger.log_info(
#                     f"Report for {model_name} generated at {report_path}"
#                 )

#         print(f"Training completed for {model_name}")
#         accuracy = metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
#         training_time = metrics.get("training_time_seconds", 0)

#         print(f"Final accuracy: {accuracy:.4f}")
#         print(f"Training time: {training_time:.2f} seconds")

#         if batch_logger:
#             batch_logger.log_info(f"Model {model_name} training successful")
#             batch_logger.log_info(f"Final accuracy: {accuracy:.4f}")
#             batch_logger.log_info(f"Training time: {training_time:.2f} seconds")

#             # Log summary metrics for the batch report
#             batch_summary = {
#                 f"{model_name}_accuracy": accuracy,
#                 f"{model_name}_training_time": training_time,
#             }
#             batch_logger.log_metrics(batch_summary)

#         return True, metrics

#     except Exception as e:
#         error_msg = f"Error training model {model_name}: {e}"
#         print(error_msg)
#         import traceback

#         trace = traceback.format_exc()
#         print(trace)

#         if batch_logger:
#             batch_logger.log_error(error_msg)
#             batch_logger.log_error(trace)

#         return False, {"error": str(e)}


# def main():
#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Plant Disease Detection Training")
#     parser.add_argument(
#         "--config",
#         type=str,
#         default=None,
#         help="Path to the configuration file",
#     )
#     parser.add_argument(
#         "--models",
#         type=str,
#         nargs="+",
#         default=None,
#         help="Model architectures to train (space-separated list)",
#     )
#     parser.add_argument(
#         "--all_models",
#         action="store_true",
#         help="Train all models defined in the configuration",
#     )
#     parser.add_argument(
#         "--data_dir", type=str, default=None, help="Path to the dataset directory"
#     )
#     parser.add_argument(
#         "--batch_size", type=int, default=None, help="Override batch size for training"
#     )
#     parser.add_argument(
#         "--epochs",
#         type=int,
#         default=None,
#         help="Override number of epochs for training",
#     )
#     args = parser.parse_args()

#     # Start timing
#     start_time = time.time()

#     # Get project paths
#     paths = get_paths()

#     # Load configuration
#     config_loader = ConfigLoader(args.config)
#     config = config_loader.get_config()

#     # Set project information
#     project_info = config.get("project", {})
#     project_name = project_info.get("name", "Plant Disease Detection")
#     project_version = project_info.get("version", "1.0.0")

#     print(f"Starting {project_name} v{project_version}")

#     # Create a directory for batch logs
#     batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     batch_log_dir = paths.logs_dir / f"batch_{batch_timestamp}"
#     batch_log_dir.mkdir(parents=True, exist_ok=True)

#     # Initialize batch logger for overall process tracking
#     batch_logger = Logger(
#         "batch_training",
#         log_dir=batch_log_dir,
#         config=config.get("logging", {}),
#         logger_type="batch",
#     )

#     batch_logger.log_info(f"Starting {project_name} v{project_version}")
#     batch_logger.log_config(config)

#     # Override configuration with command-line arguments
#     if args.epochs:
#         config["training"]["epochs"] = args.epochs
#         print(f"Overriding epochs: {args.epochs}")
#         batch_logger.log_info(f"Overriding epochs: {args.epochs}")
#     if args.batch_size:
#         config["training"]["batch_size"] = args.batch_size
#         print(f"Overriding batch size: {args.batch_size}")
#         batch_logger.log_info(f"Overriding batch size: {args.batch_size}")

#     # Configure hardware
#     configure_hardware(config)
#     batch_logger.log_hardware_metrics(step=0)

#     # Log system information
#     batch_logger.log_info("Loading datasets...")

#     # Load data (only once for all models)
#     data_loader = DataLoader(config)
#     train_data, val_data, test_data, class_names = data_loader.load_data(args.data_dir)

#     batch_logger.log_info(f"Datasets loaded with {len(class_names)} classes")
#     batch_logger.log_info(f"Classes: {list(class_names.values())}")

#     # Create model factory
#     model_factory = ModelFactory()

#     # Initialize model registry
#     registry = ModelRegistryManager()

#     # Determine which models to train
#     models_to_train = []

#     if args.all_models:
#         # Train all models in the configuration
#         models_to_train = config_loader.get_all_model_names()
#         print(f"Will train all {len(models_to_train)} models from configuration")
#         batch_logger.log_info(
#             f"Will train all {len(models_to_train)} models from configuration: {', '.join(models_to_train)}"
#         )
#     elif args.models:
#         # Train specific models
#         models_to_train = args.models
#         msg = f"Will train {len(models_to_train)} specified models: {', '.join(models_to_train)}"
#         print(msg)
#         batch_logger.log_info(msg)
#     else:
#         # Default to training a single model (ResNet50)
#         models_to_train = ["ResNet50"]
#         print("No models specified, defaulting to ResNet50")
#         batch_logger.log_info("No models specified, defaulting to ResNet50")

#     # Train all specified models
#     results = {}
#     successful_models = 0
#     failed_models = 0

#     for model_name in (model_pbar := tqdm(models_to_train, desc="Models", position=0)):
#         model_pbar.set_description(f"Training {model_name}")

#         model_start_time = time.time()
#         success, metrics = train_model(
#             model_name,
#             config,
#             data_loader,
#             model_factory,
#             train_data,
#             val_data,
#             test_data,
#             class_names,
#             batch_logger,
#         )
#         model_time = time.time() - model_start_time

#         results[model_name] = metrics
#         if success:
#             successful_models += 1
#         else:
#             failed_models += 1

#         status_str = f"{'✓' if success else '✗'} in {model_time:.1f}s"
#         model_pbar.set_postfix_str(status_str)

#         # Log model completion
#         accuracy = (
#             metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
#             if "error" not in metrics
#             else 0
#         )
#         batch_logger.log_info(
#             f"Model {model_name} completed - Status: {'Success' if success else 'Failed'}, Time: {model_time:.2f}s, Accuracy: {accuracy:.4f}"
#         )

#     # Generate comparison report if multiple models were trained
#     if len(results) > 1 and config.get("reporting", {}).get(
#         "generate_html_report", True
#     ):
#         try:
#             comparison_data = []
#             for model_name, metrics in results.items():
#                 if "error" not in metrics:
#                     comparison_data.append({"name": model_name, "metrics": metrics})

#             if comparison_data:
#                 report_generator = ReportGenerator(config)
#                 comparison_path = report_generator.generate_comparison_report(
#                     comparison_data
#                 )
#                 print(f"Model comparison report generated at {comparison_path}")
#                 batch_logger.log_info(
#                     f"Model comparison report generated at {comparison_path}"
#                 )
#         except Exception as e:
#             error_msg = f"Error generating comparison report: {e}"
#             print(error_msg)
#             batch_logger.log_error(error_msg)

#     # Print summary of all trained models
#     total_time = time.time() - start_time

#     print("\n\nTraining Summary:")
#     print("=" * 80)
#     print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
#     print(f"Models trained: {successful_models} successful, {failed_models} failed")

#     batch_logger.log_info("\nTraining Summary:")
#     batch_logger.log_info(
#         f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)"
#     )
#     batch_logger.log_info(
#         f"Models trained: {successful_models} successful, {failed_models} failed"
#     )

#     # Prepare the final metrics for the batch logger
#     batch_metrics = {
#         "total_time_seconds": total_time,
#         "total_time_minutes": total_time / 60,
#         "successful_models": successful_models,
#         "failed_models": failed_models,
#         "total_models": len(models_to_train),
#     }

#     # Create detailed model results for batch logging
#     for model_name, metrics in results.items():
#         if "error" in metrics:
#             print(f"{model_name}: Failed - {metrics['error']}")
#             batch_logger.log_info(f"{model_name}: Failed - {metrics['error']}")
#             batch_metrics[f"{model_name}_status"] = "failed"
#             batch_metrics[f"{model_name}_error"] = metrics["error"]
#         else:
#             accuracy = metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
#             train_time = metrics.get("training_time_seconds", 0)
#             print(
#                 f"{model_name}: Success - Accuracy: {accuracy:.4f} - Time: {train_time:.2f}s"
#             )
#             batch_logger.log_info(
#                 f"{model_name}: Success - Accuracy: {accuracy:.4f} - Time: {train_time:.2f}s"
#             )
#             batch_metrics[f"{model_name}_status"] = "success"
#             batch_metrics[f"{model_name}_accuracy"] = accuracy
#             batch_metrics[f"{model_name}_training_time"] = train_time

#     # Save final batch metrics
#     batch_logger.save_final_metrics(batch_metrics)

#     print("=" * 80)
#     print("Training completed.")
#     batch_logger.log_info("Training completed.")


# if __name__ == "__main__":
#     main()
