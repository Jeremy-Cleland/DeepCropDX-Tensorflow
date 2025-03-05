# src/models/attention.py
"""
Attention mechanisms and model enhancements for deep learning models.

This module provides various attention mechanisms and enhancements
that can be integrated with standard CNN architectures to improve
performance on plant disease detection tasks.
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Reshape,
    multiply,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    Activation,
    BatchNormalization,
    Lambda,
    Concatenate,
    Dropout,
    Add,
)


class SpatialAttention(tf.keras.layers.Layer):
    """Spatial Attention layer implementation"""

    def __init__(self, kernel_size=7, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.concat_layer = tf.keras.layers.Concatenate(axis=-1)
        self.multiply_layer = tf.keras.layers.Multiply()

    def build(self, input_shape):
        self.conv = tf.keras.layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            strides=1,
            padding="same",
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=False,
        )
        self.avg_pool = tf.keras.layers.Lambda(
            lambda x: tf.reduce_mean(x, axis=-1, keepdims=True),
            name="channel_avg_pool"
        )
        self.max_pool = tf.keras.layers.Lambda(
            lambda x: tf.reduce_max(x, axis=-1, keepdims=True),
            name="channel_max_pool"
        )
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        # Average pooling along channel axis
        avg_pool = self.avg_pool(inputs)
        # Max pooling along channel axis
        max_pool = self.max_pool(inputs)

        # Concatenate both features
        concat = self.concat_layer([avg_pool, max_pool])

        # Apply convolution
        spatial = self.conv(concat)

        # Apply attention
        output = self.multiply_layer([inputs, spatial])

        return output

    def get_config(self):
        config = super(SpatialAttention, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class ChannelAttention(tf.keras.layers.Layer):
    """Channel Attention layer implementation (Squeeze-and-Excitation)"""

    def __init__(self, ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio
        self.multiply_layer = tf.keras.layers.Multiply()

    def build(self, input_shape):
        self.input_rank = len(input_shape)
        channels = input_shape[-1]
        
        # For 4D inputs (with spatial dimensions)
        if self.input_rank == 4:
            self.gap = tf.keras.layers.GlobalAveragePooling2D()
            self.reshape = tf.keras.layers.Reshape((1, 1, channels))
        
        # FC layers for attention
        self.dense1 = tf.keras.layers.Dense(
            channels // self.ratio,
            activation="relu",
            kernel_initializer="he_normal",
            use_bias=True,
            bias_initializer="zeros",
        )
        self.dense2 = tf.keras.layers.Dense(
            channels,
            activation="sigmoid",
            kernel_initializer="he_normal",
            use_bias=True,
            bias_initializer="zeros",
        )
        
        super(ChannelAttention, self).build(input_shape)

    def call(self, inputs):
        # Handle different input shapes
        if self.input_rank == 4:
            # Input has spatial dimensions, apply pooling
            x = self.gap(inputs)
            
            # MLP with bottleneck
            x = self.dense1(x)
            x = self.dense2(x)
            
            # Reshape for broadcasting
            x = self.reshape(x)
            
            # Apply attention
            output = self.multiply_layer([inputs, x])
        else:
            # Input already pooled (2D), just apply FC layers
            x = self.dense1(inputs)
            x = self.dense2(x)
            
            # Element-wise multiplication for 2D inputs
            output = self.multiply_layer([inputs, x])
            
        return output

    def get_config(self):
        config = super(ChannelAttention, self).get_config()
        config.update({"ratio": self.ratio})
        return config


class CBAMBlock(tf.keras.layers.Layer):
    """Convolutional Block Attention Module (CBAM)"""

    def __init__(self, ratio=16, kernel_size=7, **kwargs):
        super(CBAMBlock, self).__init__(**kwargs)
        self.ratio = ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.input_rank = len(input_shape)
        self.channel_attention = ChannelAttention(ratio=self.ratio)
        
        # Only apply spatial attention if we have spatial dimensions
        if self.input_rank == 4:
            self.spatial_attention = SpatialAttention(kernel_size=self.kernel_size)
            
        super(CBAMBlock, self).build(input_shape)

    def call(self, inputs):
        # Channel attention
        x = self.channel_attention(inputs)

        # Spatial attention only if we have spatial dimensions
        if self.input_rank == 4:
            x = self.spatial_attention(x)

        return x

    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape
        
    def get_config(self):
        config = super(CBAMBlock, self).get_config()
        config.update({"ratio": self.ratio, "kernel_size": self.kernel_size})
        return config


class SEBlock(tf.keras.layers.Layer):
    """Squeeze-and-Excitation Block"""

    def __init__(self, ratio=16, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        self.channel_attention = ChannelAttention(ratio=self.ratio)
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        return self.channel_attention(inputs)
        
    def compute_output_shape(self, input_shape):
        # Output shape is the same as input shape
        return input_shape

    def get_config(self):
        config = super(SEBlock, self).get_config()
        config.update({"ratio": self.ratio})
        return config


def add_attention_to_model(model, attention_type="se", ratio=16, kernel_size=7):
    """Add attention mechanisms to an existing model

    Args:
        model: Input Keras model
        attention_type: Type of attention ('se', 'cbam', or 'spatial')
        ratio: Reduction ratio for the channel attention
        kernel_size: Kernel size for spatial attention

    Returns:
        New model with attention blocks
    """
    # Get the appropriate attention layer
    if attention_type == "se":
        attention_layer = SEBlock(ratio=ratio)
    elif attention_type == "cbam":
        attention_layer = CBAMBlock(ratio=ratio, kernel_size=kernel_size)
    elif attention_type == "spatial":
        attention_layer = SpatialAttention(kernel_size=kernel_size)
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")

    # Extract the base model's input and output
    inputs = model.input
    
    # Apply attention to the model's output
    x = model.output
    outputs = attention_layer(x)

    # Create a new model
    from tensorflow.keras.models import Model

    enhanced_model = Model(inputs=inputs, outputs=outputs)

    return enhanced_model


def get_attention_layer(attention_type="se", ratio=16, kernel_size=7):
    """Get the specified attention layer

    Args:
        attention_type: Type of attention ('se', 'cbam', or 'spatial')
        ratio: Reduction ratio for the channel attention
        kernel_size: Kernel size for spatial attention

    Returns:
        Attention layer
    """
    if attention_type == "se":
        return SEBlock(ratio=ratio)
    elif attention_type == "cbam":
        return CBAMBlock(ratio=ratio, kernel_size=kernel_size)
    elif attention_type == "spatial":
        return SpatialAttention(kernel_size=kernel_size)
    else:
        raise ValueError(f"Unsupported attention type: {attention_type}")


def squeeze_and_excitation_block(input_tensor, ratio=16):
    """
    Add Squeeze-and-Excitation block to any model architecture.

    Args:
        input_tensor: Input tensor to apply SE block to
        ratio: Reduction ratio for the squeeze operation

    Returns:
        Output tensor with SE applied
    """
    # Use the SEBlock layer directly
    return SEBlock(ratio=ratio)(input_tensor)


class ResidualAttention(tf.keras.layers.Layer):
    """
    Residual Attention module for enhancing ResNet-like architectures.

    This implements channel attention similar to SE blocks but with a
    residual connection to maintain gradient flow.
    """

    def __init__(self, reduction=16, **kwargs):
        """
        Initialize the Residual Attention module.

        Args:
            reduction: Reduction ratio for the bottleneck
        """
        super(ResidualAttention, self).__init__(**kwargs)
        self.reduction = reduction
        self.multiply_layer = tf.keras.layers.Multiply()

    def build(self, input_shape):
        channels = input_shape[-1]
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.dense1 = tf.keras.layers.Dense(
            channels // self.reduction, 
            activation="relu",
            kernel_initializer="he_normal"
        )
        self.dense2 = tf.keras.layers.Dense(
            channels, 
            activation="sigmoid",
            kernel_initializer="he_normal"
        )
        self.reshape = tf.keras.layers.Reshape((1, 1, channels))
        super(ResidualAttention, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass for the residual attention module.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor with attention applied
        """
        y = self.avg_pool(inputs)
        y = self.dense1(y)
        y = self.dense2(y)
        y = self.reshape(y)
        return self.multiply_layer([inputs, y])
        
    def get_config(self):
        config = super(ResidualAttention, self).get_config()
        config.update({"reduction": self.reduction})
        return config


class ECABlock(tf.keras.layers.Layer):
    """
    Efficient Channel Attention (ECA) block.

    This is a more lightweight alternative to SE blocks that uses
    1D convolutions instead of fully connected layers.
    """

    def __init__(self, kernel_size=3, **kwargs):
        """
        Initialize the ECA block.

        Args:
            kernel_size: Size of the 1D convolution kernel
        """
        super(ECABlock, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.multiply_layer = tf.keras.layers.Multiply()

    def build(self, input_shape):
        """
        Build the ECA block.

        Args:
            input_shape: Shape of the input tensor
        """
        self.channels = input_shape[-1]
        self.avg_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.conv = tf.keras.layers.Conv1D(
            filters=1, kernel_size=self.kernel_size, padding="same", use_bias=False
        )
        self.reshape_1 = tf.keras.layers.Reshape((1, self.channels))
        self.activation = tf.keras.layers.Activation('sigmoid')
        self.reshape_2 = tf.keras.layers.Reshape((1, 1, self.channels))
        super(ECABlock, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass for the ECA block.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor with ECA applied
        """
        # Global average pooling
        y = self.avg_pool(inputs)
        
        # Reshape to [batch, 1, channels]
        y = self.reshape_1(y)
        
        # Apply 1D convolution
        y = self.conv(y)
        
        # Apply sigmoid activation
        y = self.activation(y)
        
        # Reshape for broadcasting
        y = self.reshape_2(y)
        
        # Apply attention
        return self.multiply_layer([inputs, y])
        
    def get_config(self):
        config = super(ECABlock, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def spatial_attention_block(input_tensor):
    """
    Spatial Attention Block for highlighting important spatial regions.

    Args:
        input_tensor: Input tensor to apply spatial attention to

    Returns:
        Output tensor with spatial attention applied
    """
    # Use the SpatialAttention layer directly
    return SpatialAttention(kernel_size=7)(input_tensor)


def cbam_block(input_tensor, ratio=16):
    """
    Convolutional Block Attention Module (CBAM).

    This combines both channel attention (similar to SE) and spatial attention.

    Args:
        input_tensor: Input tensor to apply CBAM to
        ratio: Reduction ratio for channel attention

    Returns:
        Output tensor with CBAM applied
    """
    # Use the full CBAMBlock Keras layer instead of functional API
    cbam = CBAMBlock(ratio=ratio)(input_tensor)
    return cbam


class PyramidPoolingModule(tf.keras.layers.Layer):
    """
    Pyramid Pooling Module from PSPNet for capturing multi-scale context.

    This module helps in capturing global contextual information.
    """

    def __init__(self, pool_sizes=[1, 2, 3, 6]):
        """
        Initialize the Pyramid Pooling Module.

        Args:
            pool_sizes: List of pooling factors
        """
        super(PyramidPoolingModule, self).__init__()
        self.pool_sizes = pool_sizes

    def build(self, input_shape):
        """
        Build the Pyramid Pooling Module.

        Args:
            input_shape: Shape of the input tensor
        """
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.channels = input_shape[3]

        self.conv_layers = []
        for _ in self.pool_sizes:
            self.conv_layers.append(
                Conv2D(self.channels // 4, kernel_size=1, use_bias=False)
            )

        super(PyramidPoolingModule, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass for the Pyramid Pooling Module.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor with pyramid pooling applied
        """
        features = [inputs]

        for pool_size, conv in zip(self.pool_sizes, self.conv_layers):
            # Compute pooling size
            stride = self.height // pool_size

            # Apply pooling
            x = MaxPooling2D(pool_size=(stride, stride))(inputs)

            # Apply 1x1 convolution
            x = conv(x)

            # Upsample back to original size
            x = UpSampling2D(size=(stride, stride), interpolation="bilinear")(x)

            # Add to features list
            features.append(x)

        # Concatenate all features
        return Concatenate(axis=-1)(features)


def apply_progressive_freezing(model, num_layers_to_freeze, finetuning_epochs=5):
    """
    Implement progressive freezing strategy for transfer learning.

    This function creates a callback that gradually unfreezes deeper layers
    during training for more effective fine-tuning.

    Args:
        model: The model to apply progressive freezing to
        num_layers_to_freeze: Number of layers to keep frozen throughout training
        finetuning_epochs: Number of epochs over which to unfreeze layers

    Returns:
        Callback that unfreezes layers progressively
    """
    # First phase: Train only classifier (freeze feature extractor)
    for layer in model.layers[:-2]:
        layer.trainable = False

    # Calculate how many layers to unfreeze per epoch
    unfreeze_per_epoch = max(
        1, (len(model.layers) - num_layers_to_freeze) // finetuning_epochs
    )

    # Create a callback to unfreeze layers gradually
    def unfreeze_next_layers(epoch, logs=None):
        if epoch >= 1 and epoch <= finetuning_epochs:
            layers_to_unfreeze = epoch * unfreeze_per_epoch
            for i, layer in enumerate(model.layers):
                if (
                    i >= num_layers_to_freeze
                    and i < len(model.layers) - layers_to_unfreeze
                ):
                    layer.trainable = True
                    print(f"Unfreezing layer {i}: {layer.name}")

            # Recompile the model to make the change effective
            optimizer = model.optimizer
            loss = model.loss
            metrics = model.compiled_metrics._metrics
            model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            print(
                f"Epoch {epoch}: Unfroze {layers_to_unfreeze} layers. Model recompiled."
            )

    return tf.keras.callbacks.LambdaCallback(on_epoch_begin=unfreeze_next_layers)


def discriminative_learning_rates(model, base_lr=0.001, factor=0.3):
    """
    Apply discriminative learning rates to different model parts.

    This allows having lower learning rates for early layers and
    higher learning rates for later layers.

    Args:
        model: The model to apply discriminative learning rates to
        base_lr: Base learning rate
        factor: Multiplier factor between layer groups

    Returns:
        List of (layer, learning_rate) tuples
    """
    # Group layers into 4 sections
    total_layers = len(model.layers)
    section_size = total_layers // 4

    layer_lrs = []

    for i, layer in enumerate(model.layers):
        # Early layers: lowest learning rate
        if i < section_size:
            lr = base_lr * (factor**3)
        # Early-middle layers
        elif i < section_size * 2:
            lr = base_lr * (factor**2)
        # Middle-late layers
        elif i < section_size * 3:
            lr = base_lr * factor
        # Latest layers: highest learning rate
        else:
            lr = base_lr

        layer_lrs.append((layer, lr))

    return layer_lrs


def create_efficientnet_with_attention(
    num_classes, input_shape=(224, 224, 3), base_model_name="EfficientNetB0"
):
    """
    Create an EfficientNet model with attention mechanisms.

    Args:
        num_classes: Number of output classes
        input_shape: Input shape for the model
        base_model_name: Name of the EfficientNet variant to use

    Returns:
        EfficientNet model with attention mechanisms
    """
    # Select the base model
    if base_model_name == "EfficientNetB0":
        base_model = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif base_model_name == "EfficientNetB1":
        base_model = tf.keras.applications.EfficientNetB1(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif base_model_name == "EfficientNetB2":
        base_model = tf.keras.applications.EfficientNetB2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported model name: {base_model_name}")

    # Define where to add attention mechanisms
    # For EfficientNet, we'll add attention after specific blocks
    # These are approximate indices - may need adjustment based on model structure
    if base_model_name == "EfficientNetB0":
        attention_layers = [30, 50, 100, 130]
    elif base_model_name == "EfficientNetB1":
        attention_layers = [40, 70, 120, 160]
    elif base_model_name == "EfficientNetB2":
        attention_layers = [50, 90, 140, 190]

    # Build the model with attention
    inputs = base_model.input
    x = base_model.output

    # Add attention after specific blocks
    intermediate_outputs = []
    for i, layer in enumerate(base_model.layers):
        if i in attention_layers:
            layer_output = layer.output
            attention = squeeze_and_excitation_block(layer_output)
            intermediate_outputs.append(attention)

    # Global spatial pyramid pooling for multi-scale information
    pyramid_pooling = PyramidPoolingModule()(x)

    # Classifier head
    x = GlobalAveragePooling2D()(pyramid_pooling)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Create and return the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model


def create_resnet_with_attention(
    num_classes, input_shape=(224, 224, 3), base_model_name="ResNet50"
):
    """
    Create a ResNet model with residual attention mechanisms.

    Args:
        num_classes: Number of output classes
        input_shape: Input shape for the model
        base_model_name: Name of the ResNet variant to use

    Returns:
        ResNet model with attention mechanisms
    """
    # Select the base model
    if base_model_name == "ResNet50":
        base_model = tf.keras.applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    elif base_model_name == "ResNet101":
        base_model = tf.keras.applications.ResNet101(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        raise ValueError(f"Unsupported model name: {base_model_name}")

    # We'll use a different approach instead of iterating through layers
    # Get base model features and add attention mechanism to the end
    features = base_model.output
    
    # Add CBAM attention mechanism
    attention_features = CBAMBlock()(features)
    
    # Add classification head
    x = tf.keras.layers.GlobalAveragePooling2D()(attention_features)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    # Create and return the model
    model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)

    return model
