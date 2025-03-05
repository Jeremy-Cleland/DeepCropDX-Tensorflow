"""
Configuration manager module for handling command-line arguments and configuration loading.
This is extracted from main.py to separate configuration handling from the command-line interface.
"""

import argparse
from typing import Dict, List, Any, Optional, Tuple, Union
import sys

from src.config.config_loader import ConfigLoader
from src.utils.seed_utils import set_global_seeds
from src.utils.hardware_utils import print_hardware_summary, configure_hardware


class ConfigManager:
    """Handles command-line argument parsing and configuration loading."""

    def __init__(self):
        """Initialize the configuration manager."""
        self.config = None
        self.args = None
        self.parser = self._create_argument_parser()

    def _create_argument_parser(self) -> argparse.ArgumentParser:
        """Create the argument parser with all supported command-line arguments.
        
        Returns:
            Configured ArgumentParser instance
        """
        parser = argparse.ArgumentParser(
            description="Plant Disease Detection Training System"
        )
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
        parser.add_argument(
            "--hardware_summary",
            action="store_true",
            help="Print hardware configuration summary and exit",
        )
        parser.add_argument(
            "--use_enhanced_models",
            action="store_true",
            help="Use enhanced model factory with attention mechanisms",
        )
        parser.add_argument(
            "--attention",
            type=str,
            choices=["se", "cbam", "spatial"],
            help="Add attention mechanism to standard models",
        )
        parser.add_argument(
            "--find_lr",
            action="store_true",
            help="Run learning rate finder before training",
        )
        parser.add_argument(
            "--quantize",
            action="store_true",
            help="Enable model quantization for inference",
        )
        parser.add_argument(
            "--pruning",
            action="store_true", 
            help="Enable model pruning during training"
        )
        parser.add_argument(
            "--warmup_epochs",
            type=int,
            default=None,
            help="Number of warmup epochs for learning rate"
        )
        return parser

    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command-line arguments.
        
        Args:
            args: Command-line arguments to parse (if None, uses sys.argv)
            
        Returns:
            Parsed arguments namespace
        """
        self.args = self.parser.parse_args(args)
        return self.args

    def load_config(self) -> Dict[str, Any]:
        """Load and process configuration with command-line overrides.
        
        Returns:
            Processed configuration dictionary
            
        Raises:
            ValueError: If the configuration file cannot be loaded
        """
        # Load basic configuration
        try:
            config_loader = ConfigLoader(self.args.config)
            self.config = config_loader.get_config()
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")

        # Apply command-line overrides
        self._apply_command_line_overrides()
        
        # Set up global seeds
        seed = self.config.get("seed", 42)
        set_global_seeds(seed)

        return self.config

    def _apply_command_line_overrides(self) -> None:
        """Apply command-line argument overrides to the loaded configuration."""
        if self.args.epochs:
            self.config["training"]["epochs"] = self.args.epochs
            print(f"Overriding epochs: {self.args.epochs}")
            
        if self.args.batch_size:
            self.config["training"]["batch_size"] = self.args.batch_size
            print(f"Overriding batch size: {self.args.batch_size}")
            
        if self.args.seed:
            self.config["seed"] = self.args.seed
            print(f"Overriding random seed: {self.args.seed}")
            
        if self.args.find_lr:
            # Enable learning rate finder
            if "lr_finder" not in self.config.get("training", {}):
                self.config["training"]["lr_finder"] = {}
            self.config["training"]["lr_finder"]["enabled"] = True
            print("Enabling learning rate finder")
            
        if self.args.attention:
            self.config["training"]["attention_type"] = self.args.attention
            print(f"Using {self.args.attention} attention mechanism")
            
        if self.args.quantize:
            if "optimization" not in self.config:
                self.config["optimization"] = {}
            self.config["optimization"]["quantization"] = {
                "enabled": True,
                "method": "post_training",  # Can be post_training or during_training
                "bits": 8
            }
            print("Enabling model quantization")
            
        if self.args.pruning:
            if "optimization" not in self.config:
                self.config["optimization"] = {}
            self.config["optimization"]["pruning"] = {
                "enabled": True,
                "target_sparsity": 0.5,  # Target 50% sparsity
                "pruning_schedule": "polynomial_decay"
            }
            print("Enabling model pruning")
            
        if self.args.warmup_epochs:
            if "training" not in self.config:
                self.config["training"] = {}
            if "lr_schedule" not in self.config["training"]:
                self.config["training"]["lr_schedule"] = {}
            self.config["training"]["lr_schedule"]["warmup_epochs"] = self.args.warmup_epochs
            self.config["training"]["lr_schedule"]["enabled"] = True
            print(f"Enabling learning rate warmup for {self.args.warmup_epochs} epochs")

    def get_models_to_train(self) -> List[str]:
        """Determine which models to train based on configuration and command-line arguments.
        
        Returns:
            List of model names to train
        """
        models_to_train = []

        if self.args.all_models:
            # Train all models in the configuration
            config_loader = ConfigLoader()
            models_to_train = config_loader.get_all_model_names()
            print(f"Will train all {len(models_to_train)} models from configuration")
        elif self.args.models:
            # Train specific models
            models_to_train = self.args.models
            print(f"Will train {len(models_to_train)} specified models: {', '.join(models_to_train)}")
        else:
            # Default to training a single model (ResNet50)
            models_to_train = ["ResNet50"]
            print("No models specified, defaulting to ResNet50")
            
        return models_to_train

    def should_print_hardware_summary(self) -> bool:
        """Check if hardware summary should be printed.
        
        Returns:
            True if hardware summary should be printed, False otherwise
        """
        return self.args.hardware_summary

    def should_use_tf_data(self) -> bool:
        """Check if TensorFlow Data API should be used for data loading.
        
        Returns:
            True if TF Data API should be used, False otherwise
        """
        return self.args.use_tf_data

    def should_resume_training(self) -> bool:
        """Check if training should be resumed from checkpoints.
        
        Returns:
            True if training should be resumed, False otherwise
        """
        return self.args.resume

    def get_data_directory(self) -> Optional[str]:
        """Get the data directory path.
        
        Returns:
            Data directory path if specified, None otherwise
        """
        return self.args.data_dir

    def get_attention_type(self) -> Optional[str]:
        """Get the attention mechanism type.
        
        Returns:
            Attention mechanism type if specified, None otherwise
        """
        return self.args.attention or self.config.get("training", {}).get("attention_type")