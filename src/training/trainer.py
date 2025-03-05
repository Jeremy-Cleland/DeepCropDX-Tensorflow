# src/training/trainer.py

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
from src.utils.seed_utils import set_global_seeds
from src.training.learning_rate_scheduler import get_warmup_scheduler


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

        # Set random seeds if specified
        if "seed" in self.config:
            set_global_seeds(self.config["seed"])

    def _apply_gradient_clipping(self, optimizer, clip_norm=None, clip_value=None):
        """
        Apply gradient clipping to an optimizer in a version-compatible way

        Args:
            optimizer: The optimizer to apply clipping to
            clip_norm: Value for gradient norm clipping (global)
            clip_value: Value for gradient value clipping (per-variable)

        Returns:
            Optimizer with gradient clipping applied
        """
        if clip_norm is None and clip_value is None:
            return optimizer

        # Check TensorFlow version to determine the appropriate API
        tf_version = tuple(map(int, tf.__version__.split(".")[:2]))

        if clip_norm is not None:
            self.train_logger.log_info(
                f"Using gradient norm clipping with value {clip_norm}"
            )

            # For TensorFlow 2.11+ (new API)
            if tf_version >= (2, 11):
                try:
                    # First try the newer location
                    return optimizer.extend(
                        tf.keras.optimizers.experimental.ClipByGlobalNorm(clip_norm)
                    )
                except (AttributeError, ImportError):
                    # Fallback for different API structures
                    try:
                        # Try another potential location in newer TF versions
                        return tf.keras.optimizers.extend.ClipByGlobalNorm(clip_norm)(
                            optimizer
                        )
                    except (AttributeError, ImportError):
                        self.train_logger.log_warning(
                            f"Could not apply gradient norm clipping using newer API. "
                            f"Using older method."
                        )

            # Fallback for older TensorFlow versions or if newer methods fail
            try:
                # For older TensorFlow versions that use optimizer.clipnorm
                optimizer.clipnorm = clip_norm
                self.train_logger.log_info(
                    "Applied clipnorm to optimizer (legacy method)"
                )
                return optimizer
            except Exception as e:
                self.train_logger.log_warning(
                    f"Failed to apply gradient norm clipping: {e}. "
                    "Training will proceed without gradient clipping."
                )

        if clip_value is not None:
            self.train_logger.log_info(
                f"Using gradient value clipping with value {clip_value}"
            )

            try:
                # This works on most TensorFlow versions
                optimizer.clipvalue = clip_value
                self.train_logger.log_info("Applied clipvalue to optimizer")
            except Exception as e:
                self.train_logger.log_warning(
                    f"Failed to apply gradient value clipping: {e}. "
                    "Training will proceed without gradient clipping."
                )

        return optimizer

    def train(
        self,
        model,
        model_name,
        train_data,
        validation_data=None,
        test_data=None,
        resume=False,
        callbacks=None,
    ):
        """Train a model and save results to the trials directory

        Args:
            model: TensorFlow model to train
            model_name: Name of the model
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            test_data: Test dataset (optional)
            resume: Whether to resume training from latest checkpoint if available
            callbacks: Custom callbacks to use during training (optional)

        Returns:
            Tuple of (model, history, metrics)
        """
        # Create run directory in trials folder
        run_dir = self.paths.get_model_trial_dir(model_name)

        # Check for existing checkpoint to resume from
        start_epoch = 0
        if resume:
            checkpoint_dir = Path(run_dir) / "training" / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("*.h5"))
                if checkpoint_files:
                    # Find the latest checkpoint
                    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                    print(f"Resuming training from checkpoint: {latest_checkpoint}")
                    try:
                        model = tf.keras.models.load_model(latest_checkpoint)
                        # Extract epoch number from checkpoint filename if possible
                        try:
                            filename = os.path.basename(latest_checkpoint)
                            epoch_part = filename.split("-")[1]
                            start_epoch = int(epoch_part)
                            print(f"Resuming from epoch {start_epoch}")
                        except:
                            print(
                                "Could not determine start epoch from checkpoint filename"
                            )

                        print("Successfully loaded checkpoint")
                    except Exception as e:
                        print(f"Failed to load checkpoint: {e}")
                        print("Starting fresh training instead")
                        # Proceed with original model if checkpoint loading fails
                else:
                    print(
                        "No checkpoints found to resume from. Starting fresh training."
                    )
            else:
                print("No checkpoint directory found. Starting fresh training.")

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

        # Configure mixed precision if enabled
        mixed_precision_enabled = self.config.get("hardware", {}).get(
            "mixed_precision", True
        )
        if mixed_precision_enabled:
            self.train_logger.log_info("Enabling mixed precision training")
            try:
                from tensorflow.keras import mixed_precision

                mixed_precision.set_global_policy("mixed_float16")
                self.train_logger.log_info(
                    "Mixed precision policy set to 'mixed_float16'"
                )
            except Exception as e:
                self.train_logger.log_warning(
                    f"Failed to set mixed precision policy: {e}"
                )

        # Set up optimizer
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Wrap with LossScaleOptimizer for mixed precision if enabled
            if mixed_precision_enabled:
                try:
                    from tensorflow.keras import mixed_precision

                    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
                    self.train_logger.log_info(
                        "Using LossScaleOptimizer wrapper for mixed precision"
                    )
                except Exception as e:
                    self.train_logger.log_warning(
                        f"Failed to create LossScaleOptimizer: {e}"
                    )

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

        # Add gradient clipping if configured
        clip_norm = training_config.get("clip_norm", None)
        clip_value = training_config.get("clip_value", None)

        # Apply gradient clipping using the version-compatible helper method
        optimizer = self._apply_gradient_clipping(optimizer, clip_norm, clip_value)

        # Compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.train_logger.log_info(
            f"Model compiled with optimizer: {optimizer_name}, loss: {loss}, metrics: {metrics}"
        )

        # Set up callbacks
        if callbacks is None:
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

        # Learning rate scheduler if enabled (including new warmup schedulers)
        lr_scheduler_config = training_config.get("lr_scheduler", {})
        lr_schedule_config = training_config.get("lr_schedule", {})
        
        # Check for the new warmup scheduler
        if lr_schedule_config.get("enabled", False):
            # Create a warmup scheduler using the new implementation
            warmup_scheduler = get_warmup_scheduler(self.config)
            if warmup_scheduler:
                callbacks.append(warmup_scheduler)
                scheduler_type = lr_schedule_config.get("type", "warmup_cosine")
                self.train_logger.log_info(f"Using advanced scheduler: {scheduler_type}")
        
        # Legacy scheduler support
        elif lr_scheduler_config.get("enabled", False):
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

        # Handle class weights if enabled
        class_weights = None
        if training_config.get("class_weight") == "balanced":
            # Get class names
            class_info = getattr(train_data, "class_indices", None)
            if class_info:
                class_names = {v: k for k, v in class_info.items()}

                # Calculate class weights inversely proportional to frequency
                try:
                    if hasattr(train_data, "get_files_by_class"):
                        class_counts = [
                            len(list(train_data.get_files_by_class(c)))
                            for c in class_names.values()
                        ]
                    else:
                        # Alternative method if get_files_by_class doesn't exist
                        self.train_logger.log_warning(
                            "Dataset doesn't support get_files_by_class, estimating class distribution"
                        )
                        # Sample some batches to estimate class distribution
                        samples = []
                        for i, (_, y) in enumerate(train_data):
                            samples.append(y)
                            if i >= 10:  # Sample up to 10 batches
                                break
                        if samples:
                            y_samples = np.concatenate(samples, axis=0)
                            class_counts = np.sum(y_samples, axis=0)
                        else:
                            raise ValueError("Could not estimate class distribution")

                    # Calculate weights
                    total = sum(class_counts)
                    n_classes = len(class_names)
                    class_weights = {
                        i: total / (n_classes * count) if count > 0 else 1.0
                        for i, count in enumerate(class_counts)
                    }

                    self.train_logger.log_info(
                        f"Using balanced class weights: {class_weights}"
                    )
                except Exception as e:
                    self.train_logger.log_warning(
                        f"Failed to compute class weights: {e}. Using uniform weights."
                    )
                    class_weights = None

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
            initial_epoch=start_epoch,  # Start from the right epoch if resuming
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
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
                    from src.evaluation.metrics import calculate_metrics

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
                    from src.evaluation.visualization import (
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
