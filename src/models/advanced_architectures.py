"""
Advanced model architectures module providing support for newer model types
and techniques for plant disease detection.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Union, Callable, List, Tuple


def create_efficientnetv2(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    model_size: str = "small",
    weights: str = "imagenet",
    dropout_rate: float = 0.2,
    include_top: bool = True
) -> tf.keras.Model:
    """Create an EfficientNetV2 model.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        model_size: Size variant ('small', 'medium', 'large', 'b0', 'b1', 'b2', 'b3')
        weights: Pre-trained weights ('imagenet' or None)
        dropout_rate: Dropout rate for classification head
        include_top: Whether to include classification head
        
    Returns:
        EfficientNetV2 model
        
    Raises:
        ValueError: If model_size is not valid
    """
    # Map model size to the corresponding EfficientNetV2 variant
    size_to_model = {
        "small": tf.keras.applications.EfficientNetV2S,
        "medium": tf.keras.applications.EfficientNetV2M,
        "large": tf.keras.applications.EfficientNetV2L,
        "b0": tf.keras.applications.EfficientNetV2B0,
        "b1": tf.keras.applications.EfficientNetV2B1,
        "b2": tf.keras.applications.EfficientNetV2B2,
        "b3": tf.keras.applications.EfficientNetV2B3,
    }
    
    if model_size not in size_to_model:
        raise ValueError(
            f"Invalid model_size: {model_size}. "
            f"Must be one of: {list(size_to_model.keys())}"
        )
    
    # Get the base model
    model_class = size_to_model[model_size]
    base_model = model_class(
        include_top=False,
        weights=weights,
        input_shape=input_shape,
        pooling="avg"
    )
    
    # Create the full model
    if include_top:
        # Add classification head
        x = base_model.output
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    else:
        model = base_model
    
    return model


def create_convnext(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    model_size: str = "tiny",
    weights: str = "imagenet",
    dropout_rate: float = 0.2,
    include_top: bool = True
) -> tf.keras.Model:
    """Create a ConvNeXt model.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        model_size: Size variant ('tiny', 'small', 'base', 'large', 'xlarge')
        weights: Pre-trained weights ('imagenet' or None)
        dropout_rate: Dropout rate for classification head
        include_top: Whether to include classification head
        
    Returns:
        ConvNeXt model or None if TensorFlow version doesn't support it
    """
    try:
        # Try to import ConvNeXt (requires TensorFlow 2.9+ with keras-cv)
        # Use tf.keras.applications if available, otherwise use TF-Hub
        try:
            import tensorflow_hub as hub
            
            # Map model size to TF Hub URL
            size_to_url = {
                "tiny": "https://tfhub.dev/google/convnext/tiny/classification/1",
                "small": "https://tfhub.dev/google/convnext/small/classification/1",
                "base": "https://tfhub.dev/google/convnext/base/classification/1",
                "large": "https://tfhub.dev/google/convnext/large/classification/1",
                "xlarge": "https://tfhub.dev/google/convnext/xlarge/classification/1",
            }
            
            if model_size not in size_to_url:
                raise ValueError(
                    f"Invalid model_size: {model_size}. "
                    f"Must be one of: {list(size_to_url.keys())}"
                )
            
            # Load model from TF Hub
            hub_url = size_to_url[model_size]
            base_model = hub.KerasLayer(hub_url, trainable=True)
            
            # Create the full model
            inputs = tf.keras.layers.Input(shape=input_shape)
            
            # Preprocess input if needed
            x = inputs
            if input_shape[0] != 224 or input_shape[1] != 224:
                x = tf.keras.layers.Resizing(224, 224)(x)
            
            # Apply ConvNeXt model
            features = base_model(x)
            
            if include_top:
                # Add classification head
                if dropout_rate > 0:
                    features = tf.keras.layers.Dropout(dropout_rate)(features)
                outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
                model = tf.keras.Model(inputs=inputs, outputs=outputs)
            else:
                model = tf.keras.Model(inputs=inputs, outputs=features)
            
            return model
        
        except (ImportError, ModuleNotFoundError):
            # Try using Keras Applications directly
            # Note: This may fail on older TensorFlow versions
            from tensorflow.keras.applications import convnext
            
            # Map model size to function
            size_to_func = {
                "tiny": convnext.ConvNeXtTiny,
                "small": convnext.ConvNeXtSmall,
                "base": convnext.ConvNeXtBase,
                "large": convnext.ConvNeXtLarge,
                "xlarge": convnext.ConvNeXtXLarge,
            }
            
            if model_size not in size_to_func:
                raise ValueError(
                    f"Invalid model_size: {model_size}. "
                    f"Must be one of: {list(size_to_func.keys())}"
                )
            
            # Get the model function
            model_func = size_to_func[model_size]
            
            # Create the model
            base_model = model_func(
                include_top=False,
                weights=weights,
                input_shape=input_shape,
                pooling="avg"
            )
            
            # Create the full model
            if include_top:
                # Add classification head
                x = base_model.output
                if dropout_rate > 0:
                    x = tf.keras.layers.Dropout(dropout_rate)(x)
                outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
                model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
            else:
                model = base_model
            
            return model
    
    except Exception as e:
        print(f"Error creating ConvNeXt model: {e}")
        print("ConvNeXt models require TensorFlow 2.9+ with keras-cv or tensorflow-hub")
        return None


def create_vision_transformer(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 10,
    model_size: str = "base",
    patch_size: int = 16,
    weights: str = "imagenet21k",
    dropout_rate: float = 0.2,
    include_top: bool = True
) -> tf.keras.Model:
    """Create a Vision Transformer (ViT) model.
    
    Args:
        input_shape: Input shape (height, width, channels)
        num_classes: Number of output classes
        model_size: Size variant ('base', 'large', 'huge')
        patch_size: Patch size (16 or 32)
        weights: Pre-trained weights ('imagenet21k', 'imagenet', or None)
        dropout_rate: Dropout rate for classification head
        include_top: Whether to include classification head
        
    Returns:
        Vision Transformer model or None if TensorFlow version doesn't support it
    """
    try:
        # Try to import Vision Transformer
        # Use TF-Hub for compatibility with different TensorFlow versions
        import tensorflow_hub as hub
        
        # Map model size and patch size to TF Hub URL
        size_patch_to_url = {
            ("base", 16): "https://tfhub.dev/google/vit_b16/1",
            ("base", 32): "https://tfhub.dev/google/vit_b32/1",
            ("large", 16): "https://tfhub.dev/google/vit_l16/1",
            ("large", 32): "https://tfhub.dev/google/vit_l32/1",
            ("huge", 14): "https://tfhub.dev/google/vit_h14/1",
        }
        
        key = (model_size, patch_size)
        if key not in size_patch_to_url:
            raise ValueError(
                f"Invalid combination of model_size ({model_size}) and patch_size ({patch_size}). "
                f"Available combinations: {list(size_patch_to_url.keys())}"
            )
        
        # Load model from TF Hub
        hub_url = size_patch_to_url[key]
        vit_model = hub.KerasLayer(hub_url, trainable=True)
        
        # Create the full model
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # Preprocess input if needed
        x = inputs
        if input_shape[0] != 224 or input_shape[1] != 224:
            x = tf.keras.layers.Resizing(224, 224)(x)
        
        # Apply ViT model
        features = vit_model(x)
        
        if include_top:
            # Add classification head
            if dropout_rate > 0:
                features = tf.keras.layers.Dropout(dropout_rate)(features)
            outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(features)
            model = tf.keras.Model(inputs=inputs, outputs=outputs)
        else:
            model = tf.keras.Model(inputs=inputs, outputs=features)
        
        return model
    
    except Exception as e:
        print(f"Error creating Vision Transformer model: {e}")
        print("ViT models require tensorflow-hub")
        return None


def get_advanced_model(
    model_name: str,
    num_classes: int,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    **kwargs
) -> Optional[tf.keras.Model]:
    """Get an advanced model architecture by name.
    
    Args:
        model_name: Name of the model ('EfficientNetV2', 'ConvNeXt', 'ViT')
        num_classes: Number of output classes
        input_shape: Input shape (height, width, channels)
        **kwargs: Additional model-specific parameters
        
    Returns:
        Model instance or None if the model is not supported
        
    Raises:
        ValueError: If the model name is not valid
    """
    # Normalize model name for case-insensitive matching
    model_name_lower = model_name.lower()
    
    if "efficientnetv2" in model_name_lower:
        # Extract size from model name if provided
        if model_name_lower.endswith(("small", "medium", "large", "b0", "b1", "b2", "b3")):
            size = model_name_lower.split("efficientnetv2")[-1].strip("-_")
        else:
            size = kwargs.get("model_size", "small")
        
        return create_efficientnetv2(
            input_shape=input_shape,
            num_classes=num_classes,
            model_size=size,
            weights=kwargs.get("weights", "imagenet"),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
            include_top=kwargs.get("include_top", True)
        )
    
    elif "convnext" in model_name_lower:
        # Extract size from model name if provided
        if model_name_lower.endswith(("tiny", "small", "base", "large", "xlarge")):
            size = model_name_lower.split("convnext")[-1].strip("-_")
        else:
            size = kwargs.get("model_size", "tiny")
        
        return create_convnext(
            input_shape=input_shape,
            num_classes=num_classes,
            model_size=size,
            weights=kwargs.get("weights", "imagenet"),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
            include_top=kwargs.get("include_top", True)
        )
    
    elif any(x in model_name_lower for x in ["vit", "vision_transformer", "visiontransformer"]):
        # Extract size from model name if provided
        for size in ["base", "large", "huge"]:
            if size in model_name_lower:
                model_size = size
                break
        else:
            model_size = kwargs.get("model_size", "base")
        
        # Extract patch size from model name if provided
        for patch in ["16", "32", "14"]:
            if f"p{patch}" in model_name_lower or f"patch{patch}" in model_name_lower:
                patch_size = int(patch)
                break
        else:
            patch_size = kwargs.get("patch_size", 16)
        
        return create_vision_transformer(
            input_shape=input_shape,
            num_classes=num_classes,
            model_size=model_size,
            patch_size=patch_size,
            weights=kwargs.get("weights", "imagenet21k"),
            dropout_rate=kwargs.get("dropout_rate", 0.2),
            include_top=kwargs.get("include_top", True)
        )
    
    else:
        print(f"Unknown advanced model: {model_name}")
        return None