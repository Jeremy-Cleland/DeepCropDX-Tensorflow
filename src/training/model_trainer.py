"""
Model trainer module for handling individual model training processes.
This is extracted from main.py to separate the training logic from the command-line interface.
"""

import os
import time
import tensorflow as tf
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union, Callable

from src.config.config import get_paths
from src.utils.logger import Logger
from src.training.lr_finder import find_optimal_learning_rate
from src.utils.report_generator import ReportGenerator
from src.model_registry.registry_manager import ModelRegistryManager


def train_model(
    model_name: str,
    config: Dict[str, Any],
    data_loader: Any,
    model_factory: Any,
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    test_data: Optional[tf.data.Dataset] = None,
    class_names: Optional[Dict[int, str]] = None,
    batch_logger: Optional[Logger] = None,
    resume: bool = False,
    attention_type: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Train a single model and return results
    
    Args:
        model_name: Name of the model to train
        config: Configuration dictionary
        data_loader: DataLoader instance
        model_factory: ModelFactory instance
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset (optional)
        class_names: Dictionary mapping class indices to names
        batch_logger: Logger for batch operations (optional)
        resume: Whether to resume training from latest checkpoint
        attention_type: Type of attention mechanism to use (optional)
        
    Returns:
        Tuple of (success_flag, metrics_dict)
        
    Raises:
        ValueError: If model configuration is invalid
        RuntimeError: If an error occurs during model creation or training
    """
    print(f"\n{'='*80}")
    print(f"Training model: {model_name}")
    print(f"{'='*80}")

    if batch_logger:
        batch_logger.log_info(f"Starting training for model: {model_name}")

    try:
        # Get model hyperparameters (combining defaults with model-specific)
        from src.config.config_loader import ConfigLoader
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

        # Create model using simplified factory
        if attention_type:
            print(f"Using {model_name} with {attention_type} attention")
            if batch_logger:
                batch_logger.log_info(f"Using {model_name} with {attention_type} attention")
            model = model_factory.create_model(
                model_name=model_name, 
                num_classes=len(class_names) if class_names else None,
                attention_type=attention_type
            )
        else:
            # Get model from config (which may include attention if it's in the model name)
            print(f"Creating model {model_name} from configuration")
            if batch_logger:
                batch_logger.log_info(f"Creating model {model_name} from configuration")
            model = model_factory.get_model_from_config(
                model_name, 
                num_classes=len(class_names) if class_names else None
            )

        # Apply learning rate finder if configured
        if config.get("training", {}).get("lr_finder", {}).get("enabled", False):
            print("Running learning rate finder...")
            if batch_logger:
                batch_logger.log_info("Running learning rate finder...")

            lr_config = config.get("training", {}).get("lr_finder", {})
            min_lr = lr_config.get("min_lr", 1e-7)
            max_lr = lr_config.get("max_lr", 1.0)
            num_steps = lr_config.get("num_steps", 100)

            # Initialize optimizer for LR finder
            learning_rate = config.get("training", {}).get("learning_rate", 0.001)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Compile model temporarily for LR finder
            model.compile(
                optimizer=optimizer,
                loss=config.get("training", {}).get("loss", "categorical_crossentropy"),
                metrics=config.get("training", {}).get("metrics", ["accuracy"]),
            )

            # Create a limited dataset for LR finder
            lr_dataset = train_data.take(num_steps)

            try:
                # Run LR finder
                _, _, optimal_lr = find_optimal_learning_rate(
                    model,
                    lr_dataset,
                    optimizer=optimizer,
                    min_lr=min_lr,
                    max_lr=max_lr,
                    num_steps=num_steps,
                    plot_results=True,
                )

                # Update learning rate in configuration if requested
                if lr_config.get("use_found_lr", True):
                    print(f"Setting learning rate to optimal value: {optimal_lr:.2e}")
                    if batch_logger:
                        batch_logger.log_info(
                            f"Setting learning rate to optimal value: {optimal_lr:.2e}"
                        )
                    config["training"]["learning_rate"] = float(optimal_lr)
            except Exception as e:
                print(
                    f"Error in learning rate finder: {e}, continuing with original learning rate"
                )
                if batch_logger:
                    batch_logger.log_warning(f"Error in learning rate finder: {e}")

        # Create the trainer and train the model
        from src.training.trainer import Trainer
        trainer = Trainer(config)
        model, history, metrics = trainer.train(
            model, model_name, train_data, val_data, test_data, resume=resume
        )

        # Generate report if enabled
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

        # Clean up memory
        del model
        tf.keras.backend.clear_session()
        
        # Explicitly run garbage collection
        import gc
        gc.collect()

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

        # Clean up memory even on failure
        try:
            del model
        except:
            pass
        
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

        return False, {"error": str(e), "traceback": trace}