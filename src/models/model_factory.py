# models/model_factory.py
import tensorflow as tf

from tensorflow.keras.applications import (
    DenseNet121,
    DenseNet169,
    DenseNet201,
    ResNet50,
    ResNet101,
    ResNet152,
    EfficientNetB0,
    EfficientNetB1,
    EfficientNetB2,
    EfficientNetB3,
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
    EfficientNetB7,
    MobileNet,
    MobileNetV2,
    MobileNetV3Large,
    MobileNetV3Small,
    InceptionV3,
    InceptionResNetV2,
    Xception,
    VGG16,
    VGG19,
)

from config.config_loader import ConfigLoader


class ModelFactory:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.models_dict = {
            # ConvNeXt models
            "ConvNeXtBase": self._create_convnext_base,
            "ConvNeXtLarge": self._create_convnext_large,
            "ConvNeXtSmall": self._create_convnext_small,
            "ConvNeXtTiny": self._create_convnext_tiny,
            "ConvNeXtXLarge": self._create_convnext_xlarge,
            # DenseNet models
            "DenseNet121": DenseNet121,
            "DenseNet169": DenseNet169,
            "DenseNet201": DenseNet201,
            # EfficientNet models
            "EfficientNetB0": EfficientNetB0,
            "EfficientNetB1": EfficientNetB1,
            "EfficientNetB2": EfficientNetB2,
            "EfficientNetB3": EfficientNetB3,
            "EfficientNetB4": EfficientNetB4,
            "EfficientNetB5": EfficientNetB5,
            "EfficientNetB6": EfficientNetB6,
            "EfficientNetB7": EfficientNetB7,
            # ResNet models
            "ResNet50": ResNet50,
            "ResNet101": ResNet101,
            "ResNet152": ResNet152,
            "ResNet50V2": tf.keras.applications.ResNet50V2,
            "ResNet101V2": tf.keras.applications.ResNet101V2,
            "ResNet152V2": tf.keras.applications.ResNet152V2,
            # MobileNet models
            "MobileNet": MobileNet,
            "MobileNetV2": MobileNetV2,
            "MobileNetV3Large": MobileNetV3Large,
            "MobileNetV3Small": MobileNetV3Small,
            # Others
            "InceptionV3": InceptionV3,
            "InceptionResNetV2": InceptionResNetV2,
            "Xception": Xception,
            "VGG16": VGG16,
            "VGG19": VGG19,
        }

    def get_model(self, model_name, num_classes, input_shape=None):
        """
        Create a model with the specified name and configuration
        """
        if model_name not in self.models_dict:
            raise ValueError(
                f"Model {model_name} not supported. Available models: {', '.join(self.models_dict.keys())}"
            )

        # Load model-specific configuration
        try:
            model_config = self.config_loader.get_model_config(model_name)
            config = model_config.get(model_name, {})
        except Exception as e:
            print(
                f"Warning: Could not load config for {model_name}: {e}. Using defaults."
            )
            config = {}

        # Use provided input shape or default from config
        if input_shape is None:
            input_shape = config.get("input_shape", (224, 224, 3))

        # Get base model constructor
        model_constructor = self.models_dict[model_name]

        print(f"Creating {model_name} model...")
        # Create the base model
        if callable(model_constructor):
            try:
                base_model = model_constructor(
                    include_top=config.get("include_top", False),
                    weights=config.get("weights", "imagenet"),
                    input_shape=input_shape,
                    pooling=config.get("pooling", "avg"),
                )
                print(f"Base model created successfully")
            except Exception as e:
                print(f"Error creating base model: {e}")
                raise
        else:
            # If it's a method, call it
            base_model = model_constructor()

        # Freeze layers if fine-tuning is enabled
        if config.get("fine_tuning", {}).get("enabled", False):
            freeze_layers = config["fine_tuning"].get("freeze_layers", 0)
            for layer in base_model.layers[:freeze_layers]:
                layer.trainable = False
            print(f"Froze {freeze_layers} layers for fine-tuning")

        # Build the full model with classification head
        x = base_model.output

        # Add dropout if specified
        if config.get("dropout_rate", 0) > 0:
            dropout_rate = config["dropout_rate"]
            x = tf.keras.layers.Dropout(dropout_rate)(x)
            print(f"Added dropout with rate {dropout_rate}")

        # Add classification layer
        outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

        # Create and return the model
        model = tf.keras.models.Model(inputs=base_model.input, outputs=outputs)
        print(f"Final model created with {len(model.layers)} layers")

        return model

    # ConvNeXt models require special handling since they might not be
    # directly available in tf.keras.applications
    def _create_convnext_base(self, **kwargs):
        # Implementation depends on TensorFlow version
        try:
            return tf.keras.applications.convnext.ConvNeXtBase(**kwargs)
        except:
            # Fallback implementation if not available
            raise NotImplementedError(
                "ConvNeXtBase not available in this TensorFlow version"
            )

    def _create_convnext_large(self, **kwargs):
        try:
            return tf.keras.applications.convnext.ConvNeXtLarge(**kwargs)
        except:
            raise NotImplementedError(
                "ConvNeXtLarge not available in this TensorFlow version"
            )

    def _create_convnext_small(self, **kwargs):
        try:
            return tf.keras.applications.convnext.ConvNeXtSmall(**kwargs)
        except:
            raise NotImplementedError(
                "ConvNeXtSmall not available in this TensorFlow version"
            )

    def _create_convnext_tiny(self, **kwargs):
        try:
            return tf.keras.applications.convnext.ConvNeXtTiny(**kwargs)
        except:
            raise NotImplementedError(
                "ConvNeXtTiny not available in this TensorFlow version"
            )

    def _create_convnext_xlarge(self, **kwargs):
        try:
            return tf.keras.applications.convnext.ConvNeXtXLarge(**kwargs)
        except:
            raise NotImplementedError(
                "ConvNeXtXLarge not available in this TensorFlow version"
            )
