import os
import yaml
from pathlib import Path

from config.config import get_paths


class ConfigLoader:
    def __init__(self, config_path=None):
        """Initialize the configuration loader with an optional custom config path

        Args:
            config_path: Path to the custom configuration file. If None, uses the default.
        """
        self.paths = get_paths()

        if config_path:
            self.config_path = Path(config_path)
        else:
            self.config_path = self.paths.get_config_path()

    def get_config(self):
        """Load and return the main configuration

        Returns:
            Dictionary with configuration values, or empty dict if file not found
        """
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    config = yaml.safe_load(f)
                print(f"Loaded configuration from {self.config_path}")
                return config
            except Exception as e:
                print(f"Error loading configuration from {self.config_path}: {e}")
                return {}
        else:
            print(f"Configuration file not found at {self.config_path}")
            return {}

    def get_model_config(self, model_name):
        """
        Get configuration for a specific model.
        First checks models.yaml for all models, then falls back to individual files.

        Args:
            model_name: Name of the model to get configuration for

        Returns:
            Dictionary with model configuration

        Raises:
            ValueError: If configuration for the model is not found
        """
        # First try to get from models.yaml (centralized configs)
        models_yaml_path = self.paths.get_model_config_path()
        if models_yaml_path.exists():
            try:
                with open(models_yaml_path, "r") as f:
                    all_configs = yaml.safe_load(f)
                    if all_configs and model_name in all_configs:
                        print(
                            f"Found configuration for {model_name} in {models_yaml_path}"
                        )
                        return all_configs
            except Exception as e:
                print(
                    f"Error loading model configurations from {models_yaml_path}: {e}"
                )

        # Otherwise try model-specific file
        model_config_path = self.paths.get_model_config_path(model_name)

        # If model-specific file exists, load it
        if model_config_path.exists():
            try:
                with open(model_config_path, "r") as f:
                    model_config = yaml.safe_load(f)
                    print(
                        f"Found configuration for {model_name} in {model_config_path}"
                    )
                    return model_config
            except Exception as e:
                print(
                    f"Error loading model configuration from {model_config_path}: {e}"
                )

        # If no config found, raise an error
        raise ValueError(f"Configuration for model {model_name} not found")

    def get_all_model_names(self):
        """Get a list of all available model names from the configuration

        Returns:
            List of model names
        """
        models_yaml_path = self.paths.get_model_config_path()
        if models_yaml_path.exists():
            try:
                with open(models_yaml_path, "r") as f:
                    all_configs = yaml.safe_load(f)
                    if all_configs:
                        return list(all_configs.keys())
            except Exception as e:
                print(f"Error loading model names from {models_yaml_path}: {e}")

        return []

    def get_hyperparameters(self, model_name=None, default_config=None):
        """Get hyperparameters for training, combining default and model-specific configs

        Args:
            model_name: Name of the model to get hyperparameters for (optional)
            default_config: Default configuration to use (optional)

        Returns:
            Dictionary with hyperparameters
        """
        # Start with default config if provided, otherwise load from file
        config = default_config if default_config else self.get_config()

        # Extract training hyperparameters from main config
        hyperparams = config.get("training", {}).copy()

        # If model_name is provided, try to get model-specific hyperparameters
        if model_name:
            try:
                model_config = self.get_model_config(model_name)
                model_hyperparams = model_config.get(model_name, {}).get(
                    "hyperparameters", {}
                )

                # Merge model-specific hyperparameters (they take precedence)
                hyperparams.update(model_hyperparams)
            except Exception as e:
                print(
                    f"Warning: Could not load model-specific hyperparameters for {model_name}: {e}"
                )

        return hyperparams

    def save_config(self, config, output_path=None):
        """Save configuration to a file

        Args:
            config: Configuration dictionary to save
            output_path: Path to save the configuration to (optional)

        Returns:
            Path where the configuration was saved
        """
        if output_path is None:
            output_path = self.config_path

        # Ensure parent directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save config
        with open(output_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Configuration saved to {output_path}")
        return output_path
