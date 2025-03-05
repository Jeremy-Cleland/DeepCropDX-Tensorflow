"""
Model factory with support for standard and attention-enhanced models.
"""

import tensorflow as tf
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization

from src.config.config_loader import ConfigLoader
from src.models.attention import (
    squeeze_and_excitation_block,
    cbam_block,
    spatial_attention_block,
)


class ModelFactory:
    """A factory for creating standard and attention-enhanced models."""
    
    def __init__(self):
        """Initialize the model factory with supported models and configurations."""
        self.config_loader = ConfigLoader()
        
        # Dictionary of supported base models
        self.base_models = {
            # EfficientNet family
            "EfficientNetB0": tf.keras.applications.EfficientNetB0,
            "EfficientNetB1": tf.keras.applications.EfficientNetB1,
            "EfficientNetB2": tf.keras.applications.EfficientNetB2,
            
            # ResNet family
            "ResNet50": tf.keras.applications.ResNet50,
            "ResNet101": tf.keras.applications.ResNet101,
            
            # MobileNet family
            "MobileNet": tf.keras.applications.MobileNet,
            "MobileNetV2": tf.keras.applications.MobileNetV2,
            "MobileNetV3Small": tf.keras.applications.MobileNetV3Small,
            "MobileNetV3Large": tf.keras.applications.MobileNetV3Large,
            
            # Others
            "DenseNet121": tf.keras.applications.DenseNet121,
            "Xception": tf.keras.applications.Xception,
        }
        
        # Dictionary of attention mechanisms
        self.attention_types = {
            "se": squeeze_and_excitation_block,
            "cbam": cbam_block,
            "spatial": spatial_attention_block,
        }
    
    def create_model(self, model_name: str, num_classes: int, input_shape: tuple = (224, 224, 3), 
                     attention_type: str = None, dropout_rate: float = 0.3, 
                     freeze_layers: int = 0) -> tf.keras.Model:
        """
        Create a model with optional attention mechanism.
        
        Args:
            model_name: Name of the base model
            num_classes: Number of output classes
            input_shape: Input shape for the model (height, width, channels)
            attention_type: Type of attention to add (None, 'se', 'cbam', 'spatial')
            dropout_rate: Dropout rate for the classification head
            freeze_layers: Number of layers to freeze for transfer learning
        
        Returns:
            A configured Keras model
            
        Raises:
            ValueError: If model_name or attention_type are not supported
            ImportError: If there's an issue importing the base model
            RuntimeError: If there's an error during model creation
        """
        # Check if model is supported
        if model_name not in self.base_models:
            raise ValueError(f"Model '{model_name}' not supported. Available models: "
                            f"{', '.join(sorted(self.base_models.keys()))}")
        
        # Check if attention type is supported
        if attention_type and attention_type not in self.attention_types:
            raise ValueError(f"Attention type '{attention_type}' not supported. Available types: "
                           f"{', '.join(sorted(self.attention_types.keys()))}, or None")
        
        print(f"Creating {model_name} model...")
        
        try:
            # Create base model
            base_model = self.base_models[model_name](
                include_top=False,
                weights="imagenet",
                input_shape=input_shape,
                pooling="avg"
            )
            print(f"Base model created successfully")
            
        except ImportError as e:
            error_msg = f"Failed to import {model_name}: {str(e)}. Make sure TensorFlow version supports this model."
            print(error_msg)
            raise ImportError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Error initializing {model_name} base model: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
            
        try:
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
                x = attention_func(x)
            
            # Add classification head
            if dropout_rate > 0:
                x = Dropout(dropout_rate)(x)
                print(f"Added dropout with rate {dropout_rate}")
                
            # Final layer
            outputs = Dense(num_classes, activation="softmax")(x)
            
            # Create the model
            model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
            print(f"Final model created with {len(model.layers)} layers")
            
            return model
            
        except Exception as e:
            error_msg = f"Error assembling model architecture: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e
    
    def get_model_from_config(self, model_name: str, num_classes: int) -> tf.keras.Model:
        """
        Create a model using configuration from config files.
        
        Args:
            model_name: Name of the model
            num_classes: Number of output classes
            
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
            print(f"Warning: Could not load config for {model_name}: {str(e)}. Using defaults.")
            config = {}
        
        # Extract configuration parameters with type checking
        try:
            # Get input shape
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
            
            print(f"Loaded configuration for {model_name}: input_shape={input_shape}, "
                  f"attention_type={attention_type}, dropout_rate={dropout_rate}, "
                  f"freeze_layers={freeze_layers}")
                  
            # Create and return the model
            return self.create_model(
                model_name=base_model_name,
                num_classes=num_classes,
                input_shape=input_shape,
                attention_type=attention_type,
                dropout_rate=dropout_rate,
                freeze_layers=freeze_layers
            )
            
        except (ValueError, TypeError) as e:
            error_msg = f"Invalid configuration for {model_name}: {str(e)}"
            print(error_msg)
            raise ValueError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Error creating model from config: {str(e)}"
            print(error_msg)
            raise RuntimeError(error_msg) from e