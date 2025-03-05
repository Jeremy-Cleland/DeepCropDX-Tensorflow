"""
Learning rate scheduling utilities for training with advanced schedules.
"""

import tensorflow as tf
import numpy as np
import math
from typing import Dict, Any, Optional, Union, Callable, List


class WarmupScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler that implements warmup followed by different decay strategies."""
    
    def __init__(
        self, 
        base_lr: float = 0.001,
        warmup_epochs: int = 5, 
        total_epochs: int = 100,
        strategy: str = "cosine_decay",
        min_lr: float = 1e-6,
        verbose: int = 1
    ):
        """Initialize the learning rate scheduler with warmup.
        
        Args:
            base_lr: Base learning rate after warmup
            warmup_epochs: Number of epochs for warmup phase
            total_epochs: Total number of training epochs
            strategy: Decay strategy after warmup ('cosine_decay', 'exponential_decay', 
                      'step_decay', or 'constant')
            min_lr: Minimum learning rate
            verbose: Verbosity level
        """
        super(WarmupScheduler, self).__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.min_lr = min_lr
        self.verbose = verbose
        self.iterations = 0
        self.history = {}
        self.epochs = 0
        
    def on_epoch_begin(self, epoch, logs=None):
        """Set the learning rate for the current epoch.
        
        Args:
            epoch: Current epoch number
            logs: Training logs
        """
        self.epochs = epoch
        
        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase from 0 to base_lr
            lr = (epoch + 1) / self.warmup_epochs * self.base_lr
        else:
            # Post-warmup phase: use selected decay strategy
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            
            # Ensure progress is in [0, 1]
            progress = min(max(0.0, progress), 1.0)
            
            if self.strategy == "cosine_decay":
                # Cosine decay from base_lr to min_lr
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            elif self.strategy == "exponential_decay":
                # Exponential decay from base_lr to min_lr
                decay_rate = np.log(self.min_lr / self.base_lr)
                lr = self.base_lr * np.exp(decay_rate * progress)
            elif self.strategy == "step_decay":
                # Step decay (reduce by 1/10 at 50% and 75% of training)
                if progress >= 0.75:
                    lr = self.base_lr * 0.01
                elif progress >= 0.5:
                    lr = self.base_lr * 0.1
                else:
                    lr = self.base_lr
            else:  # "constant"
                # Constant learning rate after warmup
                lr = self.base_lr
        
        # Ensure learning rate is at least min_lr
        lr = max(self.min_lr, lr)
        
        # Set the learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        # Log the learning rate
        self.history.setdefault('lr', []).append(lr)
        
        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: LR = {lr:.2e}")
    
    def on_batch_end(self, batch, logs=None):
        """Update iterations count.
        
        Args:
            batch: Current batch number
            logs: Training logs
        """
        self.iterations += 1


class OneCycleLRScheduler(tf.keras.callbacks.Callback):
    """One-Cycle Learning Rate Policy.
    
    This implements the one-cycle learning rate policy from the paper
    "Super-Convergence: Very Fast Training of Neural Networks Using
    Large Learning Rates" by Leslie N. Smith.
    """
    
    def __init__(
        self,
        max_lr: float,
        steps_per_epoch: int,
        epochs: int,
        min_lr: float = 1e-6,
        div_factor: float = 25.0,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        verbose: int = 1
    ):
        """Initialize the one-cycle learning rate scheduler.
        
        Args:
            max_lr: Maximum learning rate
            steps_per_epoch: Number of steps (batches) per epoch
            epochs: Total number of training epochs
            min_lr: Minimum learning rate
            div_factor: Determines the initial learning rate as max_lr / div_factor
            pct_start: Percentage of the cycle where the learning rate increases
            anneal_strategy: Strategy for annealing ('cos' or 'linear')
            verbose: Verbosity level
        """
        super(OneCycleLRScheduler, self).__init__()
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.min_lr = min_lr
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.verbose = verbose
        
        # Calculate initial learning rate
        self.initial_lr = max_lr / div_factor
        
        # Calculate total steps
        self.total_steps = steps_per_epoch * epochs
        
        # Calculate steps for increasing and decreasing phases
        self.step_size_up = int(self.total_steps * pct_start)
        self.step_size_down = self.total_steps - self.step_size_up
        
        # Current step counter
        self.step_count = 0
        
        # Learning rate history
        self.history = {}
    
    def on_train_begin(self, logs=None):
        """Set initial learning rate at the start of training."""
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)
    
    def on_batch_end(self, batch, logs=None):
        """Update learning rate at the end of each batch.
        
        Args:
            batch: Current batch number
            logs: Training logs
        """
        # Calculate current learning rate
        if self.step_count <= self.step_size_up:
            # Increasing phase
            progress = self.step_count / self.step_size_up
            
            if self.anneal_strategy == "cos":
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * (1 - np.cos(np.pi * progress)) / 2
            else:  # linear
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Decreasing phase
            progress = (self.step_count - self.step_size_up) / self.step_size_down
            
            if self.anneal_strategy == "cos":
                lr = self.max_lr - (self.max_lr - self.min_lr) * (1 - np.cos(np.pi * progress)) / 2
            else:  # linear
                lr = self.max_lr - (self.max_lr - self.min_lr) * progress
        
        # Set learning rate
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        # Log the learning rate
        self.history.setdefault('lr', []).append(lr)
        
        # Log every verbose steps
        if self.verbose > 0 and self.step_count % self.verbose == 0:
            print(f"Step {self.step_count}: LR = {lr:.2e}")
        
        # Increment step counter
        self.step_count += 1


def get_warmup_scheduler(config: Dict[str, Any]) -> Optional[tf.keras.callbacks.Callback]:
    """Create a learning rate scheduler based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Learning rate scheduler callback or None if not configured
    """
    # Extract learning rate configuration
    training_config = config.get("training", {})
    lr_schedule = training_config.get("lr_schedule", {})
    
    # Check if learning rate scheduling is enabled
    if not lr_schedule.get("enabled", False):
        return None
    
    # Get schedule type
    schedule_type = lr_schedule.get("type", "warmup_cosine")
    
    # Get base learning rate
    base_lr = training_config.get("learning_rate", 0.001)
    
    # Get other parameters
    total_epochs = training_config.get("epochs", 100)
    warmup_epochs = lr_schedule.get("warmup_epochs", 5)
    min_lr = lr_schedule.get("min_lr", 1e-6)
    
    # Create scheduler based on type
    if schedule_type == "warmup_cosine":
        return WarmupScheduler(
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            strategy="cosine_decay",
            min_lr=min_lr
        )
    elif schedule_type == "warmup_exponential":
        return WarmupScheduler(
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            strategy="exponential_decay",
            min_lr=min_lr
        )
    elif schedule_type == "warmup_step":
        return WarmupScheduler(
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            strategy="step_decay",
            min_lr=min_lr
        )
    elif schedule_type == "one_cycle":
        # Calculate steps per epoch (estimate if not provided)
        steps_per_epoch = lr_schedule.get("steps_per_epoch", 100)
        max_lr = lr_schedule.get("max_lr", base_lr * 10)
        div_factor = lr_schedule.get("div_factor", 25.0)
        pct_start = lr_schedule.get("pct_start", 0.3)
        
        return OneCycleLRScheduler(
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs,
            min_lr=min_lr,
            div_factor=div_factor,
            pct_start=pct_start
        )
    else:
        # Unknown schedule type
        print(f"Unknown learning rate schedule type: {schedule_type}")
        return None