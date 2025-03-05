"""
Training pipeline module for orchestrating the model training process.
"""

import time
from typing import Dict, List, Any, Optional, Tuple

import tensorflow as tf
import gc

from src.preprocessing.data_loader import DataLoader
from src.models.model_factory import ModelFactory
from src.training.batch_trainer import BatchTrainer
from src.model_registry.registry_manager import ModelRegistryManager
from ..utils.memory_utils import optimize_memory_use


def load_datasets(
    batch_trainer: BatchTrainer, config_manager: Any, data_loader: DataLoader
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, str]]:
    """
    Load and prepare datasets for training.

    Args:
        batch_trainer: BatchTrainer instance for logging
        config_manager: Configuration manager
        data_loader: DataLoader instance

    Returns:
        Tuple containing training, validation, test datasets and class names

    Raises:
        ValueError: If no classes are found in the dataset
        Exception: For other data loading errors
    """
    batch_trainer.batch_logger.log_info("Loading datasets...")

    try:
        # Log data loading strategy
        if config_manager.should_use_tf_data():
            batch_trainer.batch_logger.log_info(
                "Using TensorFlow Data API for dataset loading"
            )
        else:
            batch_trainer.batch_logger.log_info("Using standard data loading pipeline")

        # Load datasets
        train_data, val_data, test_data, class_names = data_loader.load_data(
            config_manager.get_data_directory()
        )

        if not class_names:
            raise ValueError("No classes found in the dataset")

        batch_trainer.batch_logger.log_info(
            f"Datasets loaded with {len(class_names)} classes"
        )
        batch_trainer.batch_logger.log_info(f"Classes: {list(class_names.values())}")

        return train_data, val_data, test_data, class_names

    except ValueError as e:
        batch_trainer.batch_logger.log_error(
            f"Error loading datasets (invalid data): {str(e)}"
        )
        raise
    except Exception as e:
        batch_trainer.batch_logger.log_error(f"Error loading datasets: {str(e)}")
        raise


def execute_training_pipeline(
    config: Dict[str, Any], config_manager: Any, hardware_info: Dict[str, Any]
) -> Tuple[BatchTrainer, float, int]:
    """
    Execute the full training pipeline.

    Args:
        config: Configuration dictionary
        config_manager: Configuration manager
        hardware_info: Hardware configuration information

    Returns:
        Tuple containing the batch trainer, total training time, and exit code

    Raises:
        Exception: For any training pipeline errors
    """
    # Optimize memory at the beginning of training pipeline
    optimize_memory_use()
    logger.info("Memory optimized for training pipeline")

    # Start timing
    start_time = time.time()
    exit_code = 0

    try:
        # Set up batch trainer
        batch_trainer = BatchTrainer(config)
        batch_trainer.setup_batch_logging()

        # Get models to train
        models_to_train = config_manager.get_models_to_train()
        batch_trainer.set_models_to_train(models_to_train)

        # Log hardware configuration
        batch_trainer.batch_logger.log_info(f"Hardware configuration: {hardware_info}")
        batch_trainer.batch_logger.log_hardware_metrics(step=0)

        # Choose data loader implementation and load data
        data_loader = DataLoader(config)
        train_data, val_data, test_data, class_names = load_datasets(
            batch_trainer, config_manager, data_loader
        )

        # Initialize model factory
        model_factory = ModelFactory()
        batch_trainer.batch_logger.log_info("Initialized ModelFactory")

        # Initialize model registry
        registry = ModelRegistryManager()

        # Run batch training
        results = batch_trainer.run_batch_training(
            data_loader,
            model_factory,
            train_data,
            val_data,
            test_data,
            class_names,
            resume=config_manager.should_resume_training(),
            attention_type=config_manager.get_attention_type(),
        )

        # Generate comparison report
        batch_trainer.generate_comparison_report()

        # Calculate total time and save summary
        total_time = time.time() - start_time
        batch_trainer.save_batch_summary(total_time)

        # Clean up resources to prevent memory leaks
        batch_trainer.cleanup_resources()

    except Exception as e:
        import traceback

        print(f"Error in training pipeline: {str(e)}")
        print(traceback.format_exc())

        # Try to clean up resources even on failure
        clean_up_resources()

        exit_code = 1
        return None, 0.0, exit_code

    return batch_trainer, time.time() - start_time, exit_code


def generate_training_reports(batch_trainer: BatchTrainer, total_time: float) -> None:
    """
    Generate final reports and print summary for completed training.

    Args:
        batch_trainer: BatchTrainer instance
        total_time: Total training time in seconds
    """
    print("\n\nTraining Summary:")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(
        f"Models trained: {batch_trainer.successful_models} successful, "
        f"{batch_trainer.failed_models} failed"
    )
    print("=" * 80)
    print("Training completed.")


def clean_up_resources() -> None:
    """
    Clean up TensorFlow resources and force garbage collection.
    This helps prevent memory leaks between training runs.
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()

    # Force garbage collection
    gc.collect()

    # Additional memory cleanup if needed
    if hasattr(tf.keras.backend, "set_session"):
        tf.keras.backend.set_session(tf.compat.v1.Session())
