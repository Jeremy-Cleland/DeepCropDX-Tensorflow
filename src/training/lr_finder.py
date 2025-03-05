import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from scipy.signal import savgol_filter
import time
import logging

logger = logging.getLogger(__name__)


def find_optimal_learning_rate(
    model,
    train_dataset,
    loss_fn=None,
    optimizer=None,
    min_lr=1e-7,
    max_lr=1.0,
    num_steps=100,
    stop_factor=4.0,
    smoothing=True,
    plot_results=True,
):
    """Find optimal learning rate using exponential increase and loss tracking

    Args:
        model: The Keras model to train
        train_dataset: tf.data.Dataset containing training data
        loss_fn: Loss function to use (if None, uses model's compiled loss)
        optimizer: Optimizer to use (if None, uses model's compiled optimizer)
        min_lr: Minimum learning rate to test
        max_lr: Maximum learning rate to test
        num_steps: Number of learning rate steps to test
        stop_factor: Stop if loss exceeds best loss by this factor
        smoothing: Whether to apply smoothing to the loss curve
        plot_results: Whether to generate and display a plot

    Returns:
        Tuple of (learning_rates, losses, optimal_lr)
    """
    # Ensure dataset is batched and has at least num_steps batches
    if hasattr(train_dataset, "cardinality"):
        dataset_size = train_dataset.cardinality().numpy()
        if dataset_size == tf.data.INFINITE_CARDINALITY:
            # Dataset is repeat()-ed, we're good
            pass
        elif dataset_size < num_steps:
            logger.warning(
                f"Dataset only has {dataset_size} batches, but {num_steps} steps requested. "
                f"Creating a repeat()-ed dataset."
            )
            train_dataset = train_dataset.repeat()

    # Get loss function and optimizer from model if not provided
    if loss_fn is None:
        if not hasattr(model, "loss") or model.loss is None:
            raise ValueError(
                "No loss function provided and model is not compiled with a loss function"
            )
        loss_fn = model.loss

    if optimizer is None:
        if not hasattr(model, "optimizer") or model.optimizer is None:
            raise ValueError(
                "No optimizer provided and model is not compiled with an optimizer"
            )
        optimizer = model.optimizer

    logger.info(
        f"Starting learning rate finder from {min_lr:.1e} to {max_lr:.1e} over {num_steps} steps"
    )

    # Create a copy of model weights
    original_weights = model.get_weights()

    # Save original learning rate
    original_lr = K.get_value(optimizer.lr)

    try:
        # Exponential increase factor
        mult_factor = (max_lr / min_lr) ** (1.0 / num_steps)

        # Lists to store learning rates and losses
        learning_rates = []
        losses = []

        # Set initial learning rate
        K.set_value(optimizer.lr, min_lr)

        # Best loss tracking
        best_loss = float("inf")

        # Time tracking
        start_time = time.time()

        # Process batches
        for batch_idx, batch_data in enumerate(train_dataset):
            if batch_idx >= num_steps:
                break

            # Unpack batch data (handle different dataset formats)
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                inputs, targets = batch_data
            else:
                inputs = batch_data
                targets = None  # For models that don't need separate targets

            # Record current learning rate
            current_lr = K.get_value(optimizer.lr)
            learning_rates.append(current_lr)

            # Train on batch
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)

                # Handle different loss function signatures
                if targets is not None:
                    loss = loss_fn(targets, outputs)
                else:
                    loss = loss_fn(outputs)

            # Update weights
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Convert loss to scalar if it's a tensor
            try:
                loss_value = loss.numpy()
            except:
                loss_value = float(loss)

            # Record loss
            losses.append(loss_value)

            # Check for exploding loss
            if loss_value < best_loss:
                best_loss = loss_value

            # Print progress periodically
            if batch_idx % max(1, num_steps // 10) == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {batch_idx}/{num_steps}, lr={current_lr:.2e}, "
                    f"loss={loss_value:.4f}, elapsed={elapsed:.1f}s"
                )

            # Stop if loss is exploding
            if loss_value > stop_factor * best_loss or np.isnan(loss_value):
                logger.info(
                    f"Loss is exploding or NaN detected (loss={loss_value:.4f}, best_loss={best_loss:.4f}). "
                    f"Stopping early at step {batch_idx}/{num_steps}."
                )
                break

            # Increase learning rate
            K.set_value(optimizer.lr, current_lr * mult_factor)

        # Restore original weights and learning rate
        model.set_weights(original_weights)
        K.set_value(optimizer.lr, original_lr)

        # Convert to numpy arrays for easier manipulation
        learning_rates = np.array(learning_rates)
        losses = np.array(losses)

        # Filter out NaN and inf values
        valid_indices = ~(np.isnan(losses) | np.isinf(losses))
        learning_rates = learning_rates[valid_indices]
        losses = losses[valid_indices]

        if len(losses) == 0:
            raise ValueError(
                "No valid loss values found. All losses were NaN or infinite."
            )

        # Apply smoothing if requested
        if smoothing and len(losses) > 7:
            try:
                smoothed_losses = savgol_filter(
                    losses, min(7, len(losses) // 2 * 2 - 1), 3
                )
            except Exception as e:
                logger.warning(
                    f"Error applying smoothing filter: {e}. Using raw loss values."
                )
                smoothed_losses = losses
        else:
            smoothed_losses = losses

        # Find the point of steepest descent (minimum gradient)
        try:
            # Calculate the gradients of the loss curve
            gradients = np.gradient(smoothed_losses)

            # Find where the gradient is steepest (most negative)
            optimal_idx = np.argmin(gradients)

            # The optimal lr is typically a bit lower than the minimum gradient point
            optimal_lr = (
                learning_rates[optimal_idx] / 10.0
            )  # Division by 10 is a rule of thumb
        except Exception as e:
            logger.warning(
                f"Error finding optimal learning rate: {e}. Using fallback method."
            )
            # Fallback: Find point with fastest loss decrease
            loss_ratios = losses[1:] / losses[:-1]
            fastest_decrease_idx = np.argmin(loss_ratios)
            if fastest_decrease_idx < len(learning_rates) - 1:
                optimal_lr = learning_rates[fastest_decrease_idx]
            else:
                optimal_lr = learning_rates[len(learning_rates) // 2] / 10.0

        # Ensure we found a reasonable learning rate
        if optimal_lr <= min_lr or optimal_lr >= max_lr:
            logger.warning(
                f"Optimal learning rate ({optimal_lr:.2e}) is at or outside the bounds "
                f"of the tested range ({min_lr:.2e} - {max_lr:.2e}). Consider adjusting the range."
            )

        # Plot the results if requested
        if plot_results:
            plot_lr_finder_results(learning_rates, losses, smoothed_losses, optimal_lr)

        logger.info(
            f"Learning rate finder complete. Optimal learning rate: {optimal_lr:.2e}"
        )

        return learning_rates, losses, optimal_lr

    except Exception as e:
        # Restore original weights and learning rate in case of error
        model.set_weights(original_weights)
        K.set_value(optimizer.lr, original_lr)
        logger.error(f"Error during learning rate finding: {e}")
        raise


def plot_lr_finder_results(
    learning_rates, losses, smoothed_losses=None, optimal_lr=None
):
    """
    Plot the results of the learning rate finder.

    Args:
        learning_rates: List of learning rates
        losses: List of loss values
        smoothed_losses: List of smoothed loss values (optional)
        optimal_lr: The determined optimal learning rate (optional)
    """
    plt.figure(figsize=(12, 6))

    # Plot raw losses
    plt.plot(learning_rates, losses, "b-", alpha=0.3, label="Raw loss")

    # Plot smoothed losses if available
    if smoothed_losses is not None:
        plt.plot(learning_rates, smoothed_losses, "r-", label="Smoothed loss")

    # Mark the optimal learning rate if provided
    if optimal_lr is not None:
        plt.axvline(
            x=optimal_lr,
            color="green",
            linestyle="--",
            label=f"Optimal LR: {optimal_lr:.2e}",
        )

    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the figure
    try:
        plt.savefig("learning_rate_finder.png", dpi=300, bbox_inches="tight")
        logger.info("Learning rate finder plot saved as 'learning_rate_finder.png'")
    except Exception as e:
        logger.warning(f"Could not save learning rate finder plot: {e}")

    plt.tight_layout()
    plt.show()


def find_and_set_learning_rate(model, train_dataset, optimizer=None, **kwargs):
    """
    Find the optimal learning rate and set it in the model's optimizer.

    Args:
        model: The Keras model
        train_dataset: Training dataset
        optimizer: Optional optimizer (uses model's optimizer if None)
        **kwargs: Additional arguments to pass to find_optimal_learning_rate

    Returns:
        The optimal learning rate
    """
    # Get the optimizer from model if not provided
    if optimizer is None:
        if not hasattr(model, "optimizer") or model.optimizer is None:
            raise ValueError("Model is not compiled with an optimizer")
        optimizer = model.optimizer

    # Run the learning rate finder
    logger.info("Running learning rate finder...")
    _, _, optimal_lr = find_optimal_learning_rate(
        model, train_dataset, optimizer=optimizer, **kwargs
    )

    # Set the found learning rate
    logger.info(f"Setting optimizer learning rate to {optimal_lr:.2e}")
    K.set_value(optimizer.lr, optimal_lr)

    return optimal_lr


class LearningRateFinderCallback(tf.keras.callbacks.Callback):
    """
    Callback to find optimal learning rate before training starts.

    This callback runs the learning rate finder for one epoch before the actual training begins,
    then sets the optimal learning rate for the optimizer.
    """

    def __init__(
        self,
        min_lr=1e-7,
        max_lr=1.0,
        num_steps=100,
        stop_factor=4.0,
        use_validation=False,
        plot_results=True,
        set_lr=True,
    ):
        """
        Initialize the learning rate finder callback.

        Args:
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_steps: Number of steps for LR range test
            stop_factor: Stop if loss exceeds best loss by this factor
            use_validation: Whether to use validation data if available
            plot_results: Whether to plot the results
            set_lr: Whether to automatically set the learning rate
        """
        super(LearningRateFinderCallback, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.stop_factor = stop_factor
        self.use_validation = use_validation
        self.plot_results = plot_results
        self.set_lr = set_lr
        self.optimal_lr = None
        self.learning_rates = None
        self.losses = None

    def on_train_begin(self, logs=None):
        """Run the learning rate finder before training starts."""
        logger.info(
            "LearningRateFinderCallback: Finding optimal learning rate before training..."
        )

        # Save original learning rate
        self.original_lr = K.get_value(self.model.optimizer.lr)

        # Use validation data if available and requested
        if (
            self.use_validation
            and hasattr(self.model, "validation_data")
            and self.model.validation_data is not None
        ):
            dataset = self.model.validation_data
            logger.info("Using validation data for learning rate finder")
        else:
            dataset = self.params["train_data"]
            logger.info("Using training data for learning rate finder")

        # Run the learning rate finder
        self.learning_rates, self.losses, self.optimal_lr = find_optimal_learning_rate(
            self.model,
            dataset,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
            num_steps=self.num_steps,
            stop_factor=self.stop_factor,
            plot_results=self.plot_results,
        )

        # Set the learning rate if requested
        if self.set_lr:
            logger.info(
                f"Setting learning rate to optimal value: {self.optimal_lr:.2e}"
            )
            K.set_value(self.model.optimizer.lr, self.optimal_lr)
        else:
            # Restore original learning rate
            logger.info(f"Restoring original learning rate: {self.original_lr:.2e}")
            K.set_value(self.model.optimizer.lr, self.original_lr)

        # Log the finding
        logs = logs or {}
        logs["optimal_lr"] = self.optimal_lr

        return logs


def find_batch_aware_lr(
    model,
    train_dataset,
    batch_size_range=(16, 256),
    lr_range=(1e-6, 1e-1),
    n_batch_sizes=5,
    plot_results=True,
):
    """
    Find the optimal learning rate for different batch sizes.

    This function explores the relationship between batch size and optimal learning rate,
    which often follows a linear relationship.

    Args:
        model: The Keras model
        train_dataset: Training dataset (unbatched)
        batch_size_range: Tuple of (min_batch_size, max_batch_size)
        lr_range: Tuple of (min_lr, max_lr) for the search
        n_batch_sizes: Number of batch sizes to test
        plot_results: Whether to plot the results

    Returns:
        Dictionary mapping batch sizes to their optimal learning rates
    """
    min_batch, max_batch = batch_size_range

    # Generate batch sizes in log space
    batch_sizes = np.unique(
        np.logspace(np.log10(min_batch), np.log10(max_batch), n_batch_sizes).astype(int)
    )

    # Make sure train_dataset is unbatched
    if hasattr(train_dataset, "unbatch"):
        unbatched_dataset = train_dataset.unbatch()
    else:
        unbatched_dataset = train_dataset

    results = {}

    for batch_size in batch_sizes:
        logger.info(f"Finding optimal learning rate for batch size {batch_size}...")

        # Create a batched dataset with this batch size
        batched_dataset = unbatched_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Find the optimal learning rate
        _, _, optimal_lr = find_optimal_learning_rate(
            model,
            batched_dataset,
            min_lr=lr_range[0],
            max_lr=lr_range[1],
            plot_results=False,
        )

        results[int(batch_size)] = optimal_lr
        logger.info(f"Batch size {batch_size}: optimal LR = {optimal_lr:.2e}")

    if plot_results:
        plt.figure(figsize=(10, 6))
        batch_sizes_arr = np.array(list(results.keys()))
        lr_arr = np.array(list(results.values()))

        plt.plot(batch_sizes_arr, lr_arr, "o-", markersize=10)
        plt.xscale("log", base=2)
        plt.yscale("log", base=10)
        plt.xlabel("Batch Size (log scale)")
        plt.ylabel("Optimal Learning Rate (log scale)")
        plt.title("Batch Size vs. Optimal Learning Rate")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Linear fit in log-log space
        if len(batch_sizes_arr) >= 2:
            # Linear fit in log-log space - relationship is often LR ∝ batch_size
            coeffs = np.polyfit(np.log2(batch_sizes_arr), np.log10(lr_arr), 1)
            slope = coeffs[0]

            # Plot the fit
            x_fit = np.linspace(min(batch_sizes_arr), max(batch_sizes_arr), 100)
            y_fit = 10 ** (coeffs[0] * np.log2(x_fit) + coeffs[1])
            plt.plot(x_fit, y_fit, "r--", label=f"Fit: LR ∝ batch_size^{slope:.2f}")
            plt.legend()

        try:
            plt.savefig("batch_aware_lr.png", dpi=300, bbox_inches="tight")
            logger.info("Batch-aware learning rate plot saved as 'batch_aware_lr.png'")
        except Exception as e:
            logger.warning(f"Could not save batch-aware learning rate plot: {e}")

        plt.show()

    return results


def calculate_learning_rate(
    batch_size, results=None, reference_batch=32, reference_lr=1e-3
):
    """
    Calculate the appropriate learning rate for a given batch size based on the linear scaling rule.

    If results from find_batch_aware_lr are provided, uses interpolation from those results.
    Otherwise, uses the linear scaling rule: LR ∝ batch_size.

    Args:
        batch_size: The batch size to calculate learning rate for
        results: Optional dictionary of {batch_size: optimal_lr} from find_batch_aware_lr
        reference_batch: Reference batch size for linear scaling rule
        reference_lr: Reference learning rate for linear scaling rule

    Returns:
        The calculated learning rate for the given batch size
    """
    if results is not None and len(results) >= 2:
        # Use interpolation if we have enough data points
        batch_sizes = np.array(list(results.keys()))
        learning_rates = np.array(list(results.values()))

        # Use log-log interpolation
        log_batch_sizes = np.log2(batch_sizes)
        log_learning_rates = np.log10(learning_rates)

        # Find the interpolation coefficient (slope in log-log space)
        coeffs = np.polyfit(log_batch_sizes, log_learning_rates, 1)

        # Calculate the interpolated learning rate
        log_lr = coeffs[0] * np.log2(batch_size) + coeffs[1]
        return 10**log_lr
    else:
        # Use simple linear scaling rule: LR ∝ batch_size
        return reference_lr * (batch_size / reference_batch)


# Utility to create a function-based learning rate schedule using the finder results
def create_lr_schedule_from_finder(
    min_lr, max_lr, steps_per_epoch, epochs, warmup_epochs=5, decay_epochs=None
):
    """
    Create a learning rate schedule function based on learning rate finder results.

    Uses warmup followed by cosine decay:
    - Linear warmup from min_lr to max_lr over warmup_epochs
    - Cosine decay from max_lr to min_lr over remaining epochs

    Args:
        min_lr: Minimum learning rate (usually from learning rate finder)
        max_lr: Maximum learning rate (usually from learning rate finder)
        steps_per_epoch: Number of steps per epoch
        epochs: Total number of epochs
        warmup_epochs: Number of epochs for linear warmup
        decay_epochs: Number of epochs for decay (defaults to epochs - warmup_epochs)

    Returns:
        A function that takes (epoch, lr) and returns the new learning rate
    """
    if decay_epochs is None:
        decay_epochs = epochs - warmup_epochs

    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_schedule(epoch, lr):
        # Convert epoch to step for more granular control
        step = epoch * steps_per_epoch

        # Linear warmup phase
        if step < warmup_steps:
            return min_lr + (max_lr - min_lr) * (step / warmup_steps)

        # Cosine decay phase
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps

        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_steps))
        return min_lr + (max_lr - min_lr) * cosine_decay

    return lr_schedule


def onecycle_lr_schedule(initial_lr, max_lr, total_steps, pct_start=0.3):
    """
    Create a One-Cycle learning rate scheduler function.

    Args:
        initial_lr: Starting/ending learning rate
        max_lr: Maximum learning rate in the middle of the cycle
        total_steps: Total number of training steps
        pct_start: Percentage of cycle spent increasing LR

    Returns:
        A function for use with LearningRateScheduler callback
    """

    def schedule(step):
        # Calculate the current position in the cycle
        if step < pct_start * total_steps:
            # Increasing phase
            pct_progress = step / (pct_start * total_steps)
            return initial_lr + (max_lr - initial_lr) * pct_progress
        else:
            # Decreasing phase
            pct_progress = (step - pct_start * total_steps) / (
                (1 - pct_start) * total_steps
            )
            return max_lr - (max_lr - initial_lr) * pct_progress

    return tf.keras.callbacks.LearningRateScheduler(schedule)


class CyclicalLearningRateCallback(tf.keras.callbacks.Callback):
    """
    A callback to implement Cyclical Learning Rate policies during training.

    Supports "triangular", "triangular2", and "exp_range" policies.
    """

    def __init__(self, base_lr, max_lr, step_size, mode="triangular", gamma=0.99994):
        """
        Initialize the cyclical learning rate callback.

        Args:
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            step_size: Number of training iterations per half cycle
            mode: One of {"triangular", "triangular2", "exp_range"}
            gamma: Constant for "exp_range" mode, controls decay rate
        """
        super(CyclicalLearningRateCallback, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.iteration = 0
        self.history = {}

    def clr(self):
        """Calculate the current learning rate"""
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == "triangular2":
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                0, (1 - x)
            ) / (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                0, (1 - x)
            ) * (self.gamma**self.iteration)
        else:
            raise ValueError(
                f"Mode {self.mode} not supported. Use one of: triangular, triangular2, exp_range"
            )

        return lr

    def on_train_begin(self, logs=None):
        """Initialize at the start of training"""
        # Start from baseline LR
        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):
        """Update learning rate after each batch"""
        self.iteration += 1
        lr = self.clr()
        K.set_value(self.model.optimizer.lr, lr)

        # Store in history
        self.history.setdefault("lr", []).append(lr)
        if logs:
            self.history.setdefault("loss", []).append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs=None):
        """Log learning rate at the end of each epoch"""
        lr = K.get_value(self.model.optimizer.lr)
        logger.info(f"Epoch {epoch+1}: Cyclical learning rate = {lr:.2e}")


class AdaptiveLearningRateCallback(tf.keras.callbacks.Callback):
    """
    A callback that adapts the learning rate based on training dynamics.

    This callback monitors training metrics and adjusts the learning rate
    accordingly, reducing it when progress stalls or increasing it slightly
    when progress is consistent.
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_delta=1e-4,
        min_lr=1e-6,
        max_lr=1.0,
        increase_factor=1.05,
        increase_patience=5,
        cooldown=2,
        verbose=1,
    ):
        """
        Initialize the adaptive learning rate callback.

        Args:
            monitor: Metric to monitor
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement before reducing LR
            min_delta: Minimum change to qualify as improvement
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            increase_factor: Factor by which to increase learning rate
            increase_patience: Number of epochs with consistent improvement before increasing LR
            cooldown: Number of epochs to wait after a LR change
            verbose: Verbosity level
        """
        super(AdaptiveLearningRateCallback, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.increase_factor = increase_factor
        self.increase_patience = increase_patience
        self.cooldown = cooldown
        self.verbose = verbose

        self.cooldown_counter = 0
        self.wait = 0
        self.increase_wait = 0
        self.best = float("inf") if "loss" in monitor else -float("inf")
        self.monitor_op = np.less if "loss" in monitor else np.greater

    def on_epoch_end(self, epoch, logs=None):
        """Check metrics and adjust learning rate if needed"""
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            logger.warning(f"AdaptiveLR: {self.monitor} metric not found in logs!")
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0

        # Get current learning rate
        lr = K.get_value(self.model.optimizer.lr)

        # Check if we're better than the previous best
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.increase_wait += 1

            # Check if we should increase the learning rate
            if (
                self.increase_wait >= self.increase_patience
                and self.cooldown_counter == 0
            ):
                new_lr = min(lr * self.increase_factor, self.max_lr)

                if new_lr > lr:
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        logger.info(
                            f"Epoch {epoch+1}: AdaptiveLR increasing learning rate to {new_lr:.2e}"
                        )

                    self.cooldown_counter = self.cooldown
                    self.increase_wait = 0
        else:
            self.wait += 1
            self.increase_wait = 0

            # Check if we should decrease the learning rate
            if self.wait >= self.patience and self.cooldown_counter == 0:
                new_lr = max(lr * self.factor, self.min_lr)

                if new_lr < lr:
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        logger.info(
                            f"Epoch {epoch+1}: AdaptiveLR reducing learning rate to {new_lr:.2e}"
                        )

                    self.cooldown_counter = self.cooldown
                    self.wait = 0
