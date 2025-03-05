"""
Enhanced model factory with support for advanced architectures, attention mechanisms,
quantization, and pruning.
"""

import tensorflow as tf
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
import re

from src.config.config_loader import ConfigLoader
from src.models.attention import (
    squeeze_and_excitation_block,
    cbam_block,
    spatial_attention_block,
)
from src.models.advanced_architectures import get_advanced_model
from src.models.model_optimizer import ModelOptimizer


class ModelFactory:
    """A factory for creating and configuring models with advanced features."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the model factory with configuration.

        Args:
            config: Configuration dictionary (optional)
        """
        self.config = config or {}
        self.config_loader = ConfigLoader()
        self.model_optimizer = ModelOptimizer(config)

        # Dictionary of supported base models
        self.base_models = {
            # EfficientNet family
            "EfficientNetB0": tf.keras.applications.EfficientNetB0,
            "EfficientNetB1": tf.keras.applications.EfficientNetB1,
            "EfficientNetB2": tf.keras.applications.EfficientNetB2,
            "EfficientNetB3": tf.keras.applications.EfficientNetB3,
            "EfficientNetB4": tf.keras.applications.EfficientNetB4,
            "EfficientNetB5": tf.keras.applications.EfficientNetB5,
            "EfficientNetB6": tf.keras.applications.EfficientNetB6,
            "EfficientNetB7": tf.keras.applications.EfficientNetB7,
            # ResNet family
            "ResNet50": tf.keras.applications.ResNet50,
            "ResNet101": tf.keras.applications.ResNet101,
            "ResNet152": tf.keras.applications.ResNet152,
            "ResNet50V2": tf.keras.applications.ResNet50V2,
            "ResNet101V2": tf.keras.applications.ResNet101V2,
            "ResNet152V2": tf.keras.applications.ResNet152V2,
            # MobileNet family
            "MobileNet": tf.keras.applications.MobileNet,
            "MobileNetV2": tf.keras.applications.MobileNetV2,
            "MobileNetV3Small": tf.keras.applications.MobileNetV3Small,
            "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
            # DenseNet family
            "DenseNet121": tf.keras.applications.DenseNet121,
            "DenseNet169": tf.keras.applications.DenseNet169,
            "DenseNet201": tf.keras.applications.DenseNet201,
            # Others
            "Xception": tf.keras.applications.Xception,
            "InceptionV3": tf.keras.applications.InceptionV3,
            "InceptionResNetV2": tf.keras.applications.InceptionResNetV2,
            "NASNetMobile": tf.keras.applications.NASNetMobile,
            "NASNetLarge": tf.keras.applications.NASNetLarge,
        }

        # Dictionary of attention mechanisms
        self.attention_types = {
            "se": squeeze_and_excitation_block,
            "cbam": cbam_block,
            "spatial": spatial_attention_block,
        }

        # Try to add EfficientNetV2 models if available
        try:
            self.base_models.update(
                {
                    "EfficientNetV2S": tf.keras.applications.EfficientNetV2S,
                    "EfficientNetV2M": tf.keras.applications.EfficientNetV2M,
                    "EfficientNetV2L": tf.keras.applications.EfficientNetV2L,
                    "EfficientNetV2B0": tf.keras.applications.EfficientNetV2B0,
                    "EfficientNetV2B1": tf.keras.applications.EfficientNetV2B1,
                    "EfficientNetV2B2": tf.keras.applications.EfficientNetV2B2,
                    "EfficientNetV2B3": tf.keras.applications.EfficientNetV2B3,
                }
            )
        except AttributeError:
            # EfficientNetV2 models may not be available in earlier TensorFlow versions
            pass

    def create_model(
        self,
        model_name: str,
        num_classes: int,
        input_shape: Tuple[int, int, int] = (224, 224, 3),
        attention_type: Optional[str] = None,
        dropout_rate: float = 0.3,
        freeze_layers: int = 0,
        quantize: bool = False,
        pruning: bool = False,
        representative_dataset: Optional[Callable] = None,
    ) -> tf.keras.Model:
        """Create a model with optional attention mechanism and optimizations.

        Args:
            model_name: Name of the base model
            num_classes: Number of output classes
            input_shape: Input shape for the model (height, width, channels)
            attention_type: Type of attention to add (None, 'se', 'cbam', 'spatial')
            dropout_rate: Dropout rate for the classification head
            freeze_layers: Number of layers to freeze for transfer learning
            quantize: Whether to apply quantization
            pruning: Whether to apply pruning
            representative_dataset: Function that returns a representative dataset
                                   (required for full integer quantization)

        Returns:
            A configured Keras model

        Raises:
            ValueError: If model_name or attention_type are not supported
            ImportError: If there's an issue importing the base model
            RuntimeError: If there's an error during model creation
        """
        # Check for advanced models not in the standard list
        is_advanced_model = (
            "efficientnetv2" in model_name.lower()
            or "convnext" in model_name.lower()
            or any(
                x in model_name.lower()
                for x in ["vit", "vision_transformer", "visiontransformer"]
            )
        )

        # Check if model is supported
        if not is_advanced_model and model_name not in self.base_models:
            raise ValueError(
                f"Model '{model_name}' not supported. Available models: "
                f"{', '.join(sorted(self.base_models.keys()))}"
            )

        # Check if attention type is supported
        if attention_type and attention_type not in self.attention_types:
            raise ValueError(
                f"Attention type '{attention_type}' not supported. Available types: "
                f"{', '.join(sorted(self.attention_types.keys()))}, or None"
            )

        # Get optimization configuration from config if not specified directly
        if not quantize and self.config.get("optimization", {}).get(
            "quantization", {}
        ).get("enabled", False):
            quantize = True

        if not pruning and self.config.get("optimization", {}).get("pruning", {}).get(
            "enabled", False
        ):
            pruning = True

        try:
            # Handle advanced models
            if is_advanced_model:
                print(f"Creating advanced model: {model_name}")
                model = get_advanced_model(
                    model_name=model_name,
                    num_classes=num_classes,
                    input_shape=input_shape,
                    dropout_rate=dropout_rate,
                    include_top=True,
                )

                if model is None:
                    raise ValueError(f"Failed to create advanced model: {model_name}")
            else:
                # Create base model
                print(f"Creating standard model: {model_name}")
                base_model = self.base_models[model_name](
                    include_top=False,
                    weights="imagenet",
                    input_shape=input_shape,
                    pooling="avg",
                )

                # Freeze layers if specified
                if freeze_layers > 0:
                    for layer in base_model.layers[:freeze_layers]:
                        layer.trainable = False
                    print(f"Froze {freeze_layers} layers for fine-tuning")

                # Get output from base model
                x = base_model.output

                # Apply attention if specified
                if attention_type:
                    attention_func = self.attention_types[attention_type]
                    print(f"Adding {attention_type} attention mechanism")
                    print(f"Shape of tensor before attention: {x.shape}")
                    x = attention_func(x)

                # Add classification head
                if dropout_rate > 0:
                    x = tf.keras.layers.Dropout(dropout_rate)(x)
                    print(f"Added dropout with rate {dropout_rate}")

                # Final layer
                outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

                # Create the model
                model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

            # Apply pruning if enabled
            if pruning:
                print("Applying pruning...")
                try:
                    model = self.model_optimizer.apply_pruning(model)
                    print("Pruning applied successfully")
                except ImportError as e:
                    print(f"Pruning not applied: {e}")

            # Apply quantization if enabled
            if quantize:
                print("Applying quantization...")
                try:
                    model = self.model_optimizer.apply_quantization(
                        model=model, representative_dataset=representative_dataset
                    )
                    print("Quantization applied successfully")
                except Exception as e:
                    print(f"Quantization not applied: {e}")

            print(f"Model created successfully with {len(model.layers)} layers")
            return model

        except ImportError as e:
            error_msg = f"Failed to import {model_name}: {str(e)}. Make sure TensorFlow version supports this model."
            print(error_msg)
            raise ImportError(error_msg) from e

        except Exception as e:
            error_msg = f"Error creating model {model_name}: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

    def get_model_from_config(
        self,
        model_name: str,
        num_classes: int,
        input_shape: Optional[Tuple[int, int, int]] = None,
    ) -> tf.keras.Model:
        """Create a model using configuration from config files.

        Args:
            model_name: Name of the model
            num_classes: Number of output classes
            input_shape: Input shape for the model (optional)

        Returns:
            A configured Keras model

        Raises:
            ValueError: If the model config can't be found or is invalid
            RuntimeError: If there's an error creating the model
        """
        # Load model-specific configuration
        try:
            model_config = self.config_loader.get_model_config(model_name)
            if not model_config:
                raise ValueError(f"No configuration found for model {model_name}")

            config = model_config.get(model_name, {})
            if not config:
                raise ValueError(f"Empty configuration for model {model_name}")

        except ValueError as e:
            print(f"Warning: {str(e)}. Using defaults.")
            config = {}
        except Exception as e:
            print(
                f"Warning: Could not load config for {model_name}: {str(e)}. Using defaults."
            )
            config = {}

        # Extract configuration parameters with type checking
        try:
            # Get input shape
            if input_shape is None:
                input_shape_config = config.get("input_shape", (224, 224, 3))
                if isinstance(input_shape_config, list):
                    input_shape = tuple(input_shape_config)
                else:
                    input_shape = input_shape_config

            # Get attention type
            attention_type = config.get("attention_type", None)

            # Get dropout rate
            dropout_rate = float(config.get("dropout_rate", 0.3))

            # Get freeze layers
            fine_tuning_config = config.get("fine_tuning", {})
            if not isinstance(fine_tuning_config, dict):
                fine_tuning_config = {}
            freeze_layers = int(fine_tuning_config.get("freeze_layers", 0))

            # Get quantization and pruning settings
            quantize = (
                self.config.get("optimization", {})
                .get("quantization", {})
                .get("enabled", False)
            )
            pruning = (
                self.config.get("optimization", {})
                .get("pruning", {})
                .get("enabled", False)
            )

            # Get base model name (without attention suffix)
            base_model_name = model_name
            for suffix in ["_SE", "_CBAM", "_Attention"]:
                if model_name.endswith(suffix):
                    base_model_name = model_name.split(suffix)[0]
                    # If no attention_type specified in config, infer from suffix
                    if not attention_type:
                        if suffix == "_SE":
                            attention_type = "se"
                        elif suffix == "_CBAM":
                            attention_type = "cbam"
                        elif suffix == "_Attention":
                            attention_type = "spatial"
                    break

            print(
                f"Loaded configuration for {model_name}: input_shape={input_shape}, "
                f"attention_type={attention_type}, dropout_rate={dropout_rate}, "
                f"freeze_layers={freeze_layers}"
            )

            # Create and return the model
            return self.create_model(
                model_name=base_model_name,
                num_classes=num_classes,
                input_shape=input_shape,
                attention_type=attention_type,
                dropout_rate=dropout_rate,
                freeze_layers=freeze_layers,
                quantize=quantize,
                pruning=pruning,
            )

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid configuration for {model_name}: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg) from e

        except Exception as e:
            error_msg = f"Error creating model from config: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e

    def get_pruning_callbacks(
        self, log_dir: Optional[str] = None
    ) -> List[tf.keras.callbacks.Callback]:
        """Get callbacks needed for pruning.

        Args:
            log_dir: Directory to save pruning logs

        Returns:
            List of pruning callbacks
        """
        return self.model_optimizer.get_pruning_callbacks(log_dir=log_dir)

    def strip_pruning(self, model: tf.keras.Model) -> tf.keras.Model:
        """Remove pruning wrappers from the model for deployment.

        Args:
            model: Pruned Keras model

        Returns:
            Model with pruning configuration removed (but weights still pruned)
        """
        return self.model_optimizer.strip_pruning(model)

    def create_representative_dataset(
        self, dataset: tf.data.Dataset, num_samples: int = 100
    ) -> Callable:
        """Create a representative dataset function for quantization.

        Args:
            dataset: TensorFlow dataset to sample from
            num_samples: Number of samples to use

        Returns:
            Function that yields representative samples
        """
        return self.model_optimizer.create_representative_dataset(
            dataset=dataset, num_samples=num_samples
        )
