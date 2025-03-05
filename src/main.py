#!/usr/bin/env python3
"""
Main module for the plant disease detection model training system.
This module provides a command-line interface for training plant disease detection models.
"""

import time
import tensorflow as tf
from typing import Dict, List, Any, Optional

from src.config.config_manager import ConfigManager
from src.config.config import get_paths
from src.models.model_factory import ModelFactory
from src.preprocessing.data_loader import DataLoader
from src.training.batch_trainer import BatchTrainer
from src.utils.hardware_utils import configure_hardware, print_hardware_summary
from src.model_registry.registry_manager import ModelRegistryManager


def main() -> None:
    """Main entry point for the plant disease detection training system."""
    # Set up configuration manager
    config_manager = ConfigManager()
    args = config_manager.parse_args()

    # Print hardware summary and exit if requested
    if config_manager.should_print_hardware_summary():
        print_hardware_summary()
        return

    # Start timing
    start_time = time.time()

    try:
        # Load configuration with command-line overrides
        config = config_manager.load_config()

        # Configure hardware
        hardware_info = configure_hardware(config)

        # Set up batch trainer
        batch_trainer = BatchTrainer(config)
        batch_trainer.setup_batch_logging()

        # Get models to train
        models_to_train = config_manager.get_models_to_train()
        batch_trainer.set_models_to_train(models_to_train)

        # Get basic project info
        project_info = config.get("project", {})
        project_name = project_info.get("name", "Plant Disease Detection")
        project_version = project_info.get("version", "1.0.0")
        print(f"Starting {project_name} v{project_version} Batch Training")

        # Log hardware configuration
        batch_trainer.batch_logger.log_info(f"Hardware configuration: {hardware_info}")
        batch_trainer.batch_logger.log_hardware_metrics(step=0)
        batch_trainer.batch_logger.log_info("Loading datasets...")

        # Load datasets
        try:
            # Choose data loader implementation
            data_loader = DataLoader(config)

            if config_manager.should_use_tf_data():
                batch_trainer.batch_logger.log_info("Using TensorFlow Data API for dataset loading")
            else:
                batch_trainer.batch_logger.log_info("Using standard data loading pipeline")

            # Load datasets
            train_data, val_data, test_data, class_names = data_loader.load_data(
                config_manager.get_data_directory()
            )

            if not class_names:
                raise ValueError("No classes found in the dataset")

            batch_trainer.batch_logger.log_info(f"Datasets loaded with {len(class_names)} classes")
            batch_trainer.batch_logger.log_info(f"Classes: {list(class_names.values())}")
        except Exception as e:
            batch_trainer.batch_logger.log_error(f"Error loading datasets: {str(e)}")
            raise

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

        # Generate comparison report if needed
        batch_trainer.generate_comparison_report()

        # Calculate total time and save summary
        total_time = time.time() - start_time
        batch_trainer.save_batch_summary(total_time)

        # Print final summary
        print("\n\nTraining Summary:")
        print("=" * 80)
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        print(
            f"Models trained: {batch_trainer.successful_models} successful, "
            f"{batch_trainer.failed_models} failed"
        )
        print("=" * 80)
        print("Training completed.")

    except Exception as e:
        import traceback
        print(f"Error in main process: {str(e)}")
        print(traceback.format_exc())
        return 1

    # Clean up resources at the end
    tf.keras.backend.clear_session()
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
