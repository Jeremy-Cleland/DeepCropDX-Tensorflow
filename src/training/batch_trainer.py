"""
Batch trainer module for running multiple model training sessions.
This is extracted from main.py to separate batch training logic from the command-line interface.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm.auto import tqdm
import tensorflow as tf

from src.config.config import get_paths
from src.utils.logger import Logger
from src.utils.report_generator import ReportGenerator
from src.training.model_trainer import train_model


class BatchTrainer:
    """Handles batch training of multiple models with logging and reporting."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the batch trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = get_paths()
        self.batch_logger = None
        self.models_to_train = []
        self.results = {}
        self.successful_models = 0
        self.failed_models = 0

    def setup_batch_logging(self) -> None:
        """Set up batch logging with a timestamp-based directory."""
        # Create a directory for batch logs
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_log_dir = self.paths.logs_dir / f"batch_{batch_timestamp}"
        batch_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize batch logger for overall process tracking
        self.batch_logger = Logger(
            "batch_training",
            log_dir=batch_log_dir,
            config=self.config.get("logging", {}),
            logger_type="batch",
        )

        project_info = self.config.get("project", {})
        project_name = project_info.get("name", "Plant Disease Detection")
        project_version = project_info.get("version", "1.0.0")
        
        self.batch_logger.log_info(f"Starting {project_name} v{project_version}")
        self.batch_logger.log_config(self.config)

    def set_models_to_train(self, models: List[str]) -> None:
        """Set the list of models to train in this batch.
        
        Args:
            models: List of model names to train
        """
        self.models_to_train = models
        if self.batch_logger:
            self.batch_logger.log_info(
                f"Will train {len(models)} models: {', '.join(models)}"
            )

    def run_batch_training(
        self, 
        data_loader: Any,
        model_factory: Any,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        test_data: Optional[tf.data.Dataset],
        class_names: Dict[int, str],
        resume: bool = False,
        attention_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run batch training for all specified models.
        
        Args:
            data_loader: DataLoader instance
            model_factory: ModelFactory instance 
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset (optional)
            class_names: Dictionary mapping class indices to names
            resume: Whether to resume training from latest checkpoint
            attention_type: Type of attention mechanism to use (optional)
            
        Returns:
            Dictionary of results for each model
        """
        start_time = time.time()
        self.results = {}
        self.successful_models = 0
        self.failed_models = 0

        # Train all specified models
        for model_name in (model_pbar := tqdm(self.models_to_train, desc="Models", position=0)):
            model_pbar.set_description(f"Training {model_name}")

            model_start_time = time.time()
            
            # Clear memory before training each model
            tf.keras.backend.clear_session()
            import gc
            gc.collect()
            
            success, metrics = train_model(
                model_name,
                self.config,
                data_loader,
                model_factory,
                train_data,
                val_data,
                test_data,
                class_names,
                self.batch_logger,
                resume=resume,
                attention_type=attention_type,
            )
            
            model_time = time.time() - model_start_time

            self.results[model_name] = metrics
            if success:
                self.successful_models += 1
            else:
                self.failed_models += 1

            status_str = f"{'✓' if success else '✗'} in {model_time:.1f}s"
            model_pbar.set_postfix_str(status_str)

            # Log model completion
            accuracy = (
                metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
                if "error" not in metrics
                else 0
            )
            self.batch_logger.log_info(
                f"Model {model_name} completed - Status: {'Success' if success else 'Failed'}, "
                f"Time: {model_time:.2f}s, Accuracy: {accuracy:.4f}"
            )

        return self.results

    def generate_comparison_report(self) -> Optional[str]:
        """Generate a comparison report for all successfully trained models.
        
        Returns:
            Path to the generated report, or None if no report was generated
        """
        # Don't generate a report if there's only one model or no successful models
        if len(self.results) <= 1 or self.successful_models == 0:
            return None
            
        if not self.config.get("reporting", {}).get("generate_html_report", True):
            return None
            
        try:
            comparison_data = []
            for model_name, metrics in self.results.items():
                if "error" not in metrics:
                    comparison_data.append({"name": model_name, "metrics": metrics})

            if comparison_data:
                report_generator = ReportGenerator(self.config)
                comparison_path = report_generator.generate_comparison_report(
                    comparison_data
                )
                print(f"Model comparison report generated at {comparison_path}")
                if self.batch_logger:
                    self.batch_logger.log_info(
                        f"Model comparison report generated at {comparison_path}"
                    )
                return comparison_path
                
        except Exception as e:
            error_msg = f"Error generating comparison report: {e}"
            print(error_msg)
            if self.batch_logger:
                self.batch_logger.log_error(error_msg)
                
        return None

    def save_batch_summary(self, total_time: float) -> None:
        """Save batch training summary metrics.
        
        Args:
            total_time: Total time spent on batch training in seconds
        """
        batch_metrics = {
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "successful_models": self.successful_models,
            "failed_models": self.failed_models,
            "total_models": len(self.models_to_train),
            "seed": self.config.get("seed", 42),
        }

        # Create detailed model results for batch logging
        for model_name, metrics in self.results.items():
            if "error" in metrics:
                print(f"{model_name}: Failed - {metrics['error']}")
                self.batch_logger.log_info(f"{model_name}: Failed - {metrics['error']}")
                batch_metrics[f"{model_name}_status"] = "failed"
                batch_metrics[f"{model_name}_error"] = metrics["error"]
            else:
                accuracy = metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
                train_time = metrics.get("training_time_seconds", 0)
                print(
                    f"{model_name}: Success - Accuracy: {accuracy:.4f} - Time: {train_time:.2f}s"
                )
                self.batch_logger.log_info(
                    f"{model_name}: Success - Accuracy: {accuracy:.4f} - Time: {train_time:.2f}s"
                )
                batch_metrics[f"{model_name}_status"] = "success"
                batch_metrics[f"{model_name}_accuracy"] = accuracy
                batch_metrics[f"{model_name}_training_time"] = train_time

        # Save final batch metrics
        self.batch_logger.save_final_metrics(batch_metrics)