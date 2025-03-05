# src/scripts/train.py
"""
Train a model on a dataset with enhanced features
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
from pathlib import Path
import time

from src.config.config import get_paths
from src.config.config_loader import ConfigLoader
from src.preprocessing.data_loader import DataLoader
from src.models.model_factory import ModelFactory
from src.training.trainer import Trainer
from src.training.lr_finder import (
    find_optimal_learning_rate,
    LearningRateFinderCallback,
)

from src.utils.seed_utils import set_global_seeds
from src.utils.hardware_utils import configure_hardware, print_hardware_summary
from src.model_registry.registry_manager import ModelRegistryManager


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model with enhanced features")
    parser.add_argument(
        "--model", type=str, required=True, help="Model architecture to train"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to dataset directory"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--find_lr", action="store_true", help="Run learning rate finder"
    )
    parser.add_argument(
        "--attention", type=str, default=None, help="Attention type (se, cbam, spatial)"
    )
    parser.add_argument(
        "--use_enhanced",
        action="store_true",
        help="Use enhanced model variants like ResNet50_CBAM",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--hardware_summary",
        action="store_true",
        help="Print hardware configuration summary and exit",
    )
    args = parser.parse_args()

    # Print hardware summary if requested
    if args.hardware_summary:
        print_hardware_summary()
        return

    # Get project paths
    paths = get_paths()

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()

    # Override config with command-line arguments
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.find_lr:
        if "lr_finder" not in config.get("training", {}):
            config["training"]["lr_finder"] = {}
        config["training"]["lr_finder"]["enabled"] = True
    if args.attention:
        config["training"]["attention_type"] = args.attention
    if args.seed:
        config["seed"] = args.seed

    # Set project information
    project_info = config.get("project", {})
    project_name = project_info.get("name", "Plant Disease Detection")
    project_version = project_info.get("version", "1.0.0")

    print(f"Starting {project_name} v{project_version} Training")

    # Set random seed
    seed = config.get("seed", 42)
    set_global_seeds(seed)

    # Configure hardware
    hardware_info = configure_hardware(config)

    # Load data
    data_loader = DataLoader(config)
    train_data, val_data, test_data, class_names = data_loader.load_data(args.data_dir)

    print(f"Loaded dataset with {len(class_names)} classes")

    # Create enhanced model factory
    model_factory = ModelFactory()

    # Get model architecture
    model_name = args.model
    num_classes = len(class_names)

    print(f"Creating model: {model_name}")

    # Determine whether to use attention or an enhanced model variant
    if args.use_enhanced:
        # Use a pre-configured enhanced model variant (like ResNet50_CBAM)
        # The model_name should be one of the enhanced model names
        model = model_factory.get_model(model_name, num_classes)
    else:
        # Get standard model with optional attention type
        attention_type = args.attention or config.get("training", {}).get(
            "attention_type", None
        )
        model = model_factory.get_model(
            model_name, num_classes, attention_type=attention_type
        )

    # Get training parameters
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 32)
    epochs = training_config.get("epochs", 50)
    learning_rate = training_config.get("learning_rate", 0.001)
    optimizer_name = training_config.get("optimizer", "adam").lower()

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
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Apply discriminative learning rates if configured
    if config.get("training", {}).get("discriminative_lr", {}).get("enabled", False):
        print("Applying discriminative learning rates")
        discriminative_config = config.get("training", {}).get("discriminative_lr", {})
        base_lr = discriminative_config.get("base_lr", learning_rate)
        factor = discriminative_config.get("factor", 0.3)

        # Use the factory method to get layer-specific learning rates
        layer_lrs = model_factory.get_discriminative_learning_rates(
            model, base_lr=base_lr, factor=factor
        )

        # Log the learning rates for different layers
        print(
            f"Using discriminative learning rates with base_lr={base_lr}, factor={factor}"
        )
        print(
            f"Layer learning rates range from {min([lr for _, lr in layer_lrs])} to {max([lr for _, lr in layer_lrs])}"
        )

    # Run learning rate finder if configured
    if config.get("training", {}).get("lr_finder", {}).get("enabled", True):
        print("Running learning rate finder...")
        lr_config = config.get("training", {}).get("lr_finder", {})

        # Get parameters for LR finder
        min_lr = lr_config.get("min_lr", 1e-7)
        max_lr = lr_config.get("max_lr", 1.0)
        num_steps = lr_config.get("num_steps", 100)

        # Compile model temporarily for LR finder
        model.compile(
            optimizer=optimizer,
            loss=training_config.get("loss", "categorical_crossentropy"),
            metrics=training_config.get("metrics", ["accuracy"]),
        )

        # Create a limited dataset for LR finder
        lr_dataset = train_data.take(num_steps)

        # Run LR finder
        try:
            _, _, optimal_lr = find_optimal_learning_rate(
                model,
                lr_dataset,
                optimizer=optimizer,
                min_lr=min_lr,
                max_lr=max_lr,
                num_steps=num_steps,
                plot_results=True,
            )

            # Update learning rate in optimizer if configured
            if lr_config.get("use_found_lr", True):
                K.set_value(optimizer.lr, optimal_lr)
                print(f"Setting learning rate to optimal value: {optimal_lr:.2e}")
                # Update the config as well
                config["training"]["learning_rate"] = float(optimal_lr)
        except Exception as e:
            print(f"Error running learning rate finder: {e}")
            print("Continuing with original learning rate")

    # Create run directory for this training
    run_dir = paths.get_model_trial_dir(model_name)
    print(f"Training results will be saved to: {run_dir}")

    # Initialize trainer and train the model
    trainer = Trainer(config)

    # Define additional callbacks
    callbacks = []

    # Add progressive freezing callback if configured
    if config.get("training", {}).get("progressive_freezing", {}).get("enabled", False):
        print("Using progressive layer freezing during training")
        freeze_config = config.get("training", {}).get("progressive_freezing", {})
        freeze_layers = freeze_config.get("freeze_layers", 100)
        finetuning_epochs = freeze_config.get("finetuning_epochs", 5)

        # Use the factory method to create the progressive freezing callback
        progressive_callback = model_factory.get_progressive_freezing_callback(
            model,
            num_layers_to_freeze=freeze_layers,
            finetuning_epochs=finetuning_epochs,
        )
        callbacks.append(progressive_callback)

    start_time = time.time()

    try:
        # Train the model
        model, history, metrics = trainer.train(
            model,
            model_name,
            train_data,
            val_data,
            test_data,
            resume=args.resume,
            callbacks=callbacks,
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Register model in the registry
        try:
            registry = ModelRegistryManager()
            registry.register_model(model, model_name, metrics, history, run_dir)
            print(f"Model registered in registry")
        except Exception as e:
            print(f"Failed to register model in registry: {e}")

        # Print final metrics
        print(f"\nTraining Summary:")
        print(f"  Model: {model_name}")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Final accuracy: {metrics.get('test_accuracy', 0):.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
