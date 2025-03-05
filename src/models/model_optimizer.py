"""
Model optimization module for quantization, pruning, and other optimizations.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, Any, Optional, Union, Callable, List, Tuple
import tempfile
import os


class ModelOptimizer:
    """Handles model optimization techniques like quantization and pruning."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize model optimizer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.optimization_config = self.config.get("optimization", {})
        
    def apply_quantization(
        self, 
        model: tf.keras.Model, 
        representative_dataset: Optional[Callable] = None,
        method: str = "post_training",
        quantization_bits: int = 8
    ) -> tf.keras.Model:
        """Apply quantization to a model to reduce size and improve inference speed.
        
        Args:
            model: Keras model to quantize
            representative_dataset: Function that returns a representative dataset
                                   (required for full integer quantization)
            method: Quantization method ('post_training' or 'during_training')
            quantization_bits: Bit width for quantization (8 or 16)
            
        Returns:
            Quantized model
            
        Raises:
            ValueError: If the quantization method is not supported
        """
        # Get parameters from config if provided
        method = self.optimization_config.get("quantization", {}).get("method", method)
        quantization_bits = self.optimization_config.get("quantization", {}).get("bits", quantization_bits)
        
        if method == "post_training":
            return self._apply_post_training_quantization(model, representative_dataset, quantization_bits)
        elif method == "during_training":
            return self._apply_during_training_quantization(model, quantization_bits)
        else:
            raise ValueError(f"Unsupported quantization method: {method}. Use 'post_training' or 'during_training'.")
    
    def _apply_post_training_quantization(
        self, 
        model: tf.keras.Model, 
        representative_dataset: Optional[Callable] = None,
        quantization_bits: int = 8
    ) -> tf.keras.Model:
        """Apply post-training quantization to a model.
        
        Args:
            model: Keras model to quantize
            representative_dataset: Function that returns a representative dataset
            quantization_bits: Bit width for quantization
            
        Returns:
            Quantized model
        """
        # We need to save and reload the model for TFLite conversion
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "model.h5")
            model.save(model_path)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            if representative_dataset is not None:
                # Full integer quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = representative_dataset
                
                if quantization_bits == 8:
                    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                    converter.inference_input_type = tf.int8
                    converter.inference_output_type = tf.int8
            else:
                # Post-training dynamic range quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            
            # Save the TFLite model temporarily
            tflite_path = os.path.join(temp_dir, "model.tflite")
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
                
            # Load the TFLite model
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            # Create a quantized model with the same API as the original model
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Create a wrapper model with the same API as the original
            class QuantizedModelWrapper(tf.keras.Model):
                def __init__(self, interpreter, input_details, output_details, original_model):
                    super(QuantizedModelWrapper, self).__init__()
                    self.interpreter = interpreter
                    self.input_details = input_details
                    self.output_details = output_details
                    self.original_model = original_model
                    
                def call(self, inputs, training=False):
                    if training:
                        return self.original_model(inputs, training=True)
                    
                    # Process input data
                    self.interpreter.set_tensor(self.input_details[0]['index'], inputs)
                    
                    # Run inference
                    self.interpreter.invoke()
                    
                    # Get output
                    output = self.interpreter.get_tensor(self.output_details[0]['index'])
                    return output
                
                def get_config(self):
                    return self.original_model.get_config()
            
            # Create and return the wrapper model
            quantized_model = QuantizedModelWrapper(
                interpreter=interpreter,
                input_details=input_details,
                output_details=output_details,
                original_model=model
            )
            
            return quantized_model
    
    def _apply_during_training_quantization(
        self, 
        model: tf.keras.Model, 
        quantization_bits: int = 8
    ) -> tf.keras.Model:
        """Apply quantization-aware training to a model.
        
        Args:
            model: Keras model to apply quantization-aware training to
            quantization_bits: Bit width for quantization
            
        Returns:
            Model with quantization layers for training
        """
        # Apply TensorFlow's quantize_model function
        try:
            # Try to use TensorFlow's built-in quantization-aware training
            import tensorflow_model_optimization as tfmot
            
            quantize_model = tfmot.quantization.keras.quantize_model
            
            # Create a quantization config
            if quantization_bits == 8:
                quantization_config = tfmot.quantization.keras.QuantizationConfig(
                    activation_quantizer=tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                        num_bits=8, symmetric=False
                    ),
                    weight_quantizer=tfmot.quantization.keras.quantizers.LastValueQuantizer(
                        num_bits=8, symmetric=True
                    )
                )
            else:  # 16-bit
                quantization_config = tfmot.quantization.keras.QuantizationConfig(
                    activation_quantizer=tfmot.quantization.keras.quantizers.MovingAverageQuantizer(
                        num_bits=16, symmetric=False
                    ),
                    weight_quantizer=tfmot.quantization.keras.quantizers.LastValueQuantizer(
                        num_bits=16, symmetric=True
                    )
                )
            
            # Apply quantization to the model
            quantized_model = quantize_model(model, quantization_config)
            
            return quantized_model
            
        except (ImportError, ModuleNotFoundError):
            # If TensorFlow Model Optimization is not available, return the original model
            print("TensorFlow Model Optimization is not installed. Using original model.")
            return model
    
    def apply_pruning(
        self, 
        model: tf.keras.Model, 
        target_sparsity: float = 0.5,
        pruning_schedule: str = "polynomial_decay"
    ) -> tf.keras.Model:
        """Apply weight pruning to a model to reduce size and improve inference speed.
        
        Args:
            model: Keras model to prune
            target_sparsity: Target sparsity (percentage of weights to prune)
            pruning_schedule: Type of pruning schedule to use
            
        Returns:
            Model with pruning configuration for training
            
        Raises:
            ImportError: If TensorFlow Model Optimization is not installed
        """
        # Get parameters from config if provided
        target_sparsity = self.optimization_config.get("pruning", {}).get("target_sparsity", target_sparsity)
        pruning_schedule = self.optimization_config.get("pruning", {}).get("pruning_schedule", pruning_schedule)
        
        try:
            # Import TensorFlow Model Optimization
            import tensorflow_model_optimization as tfmot
            
            # Set up pruning params based on schedule type
            if pruning_schedule == "polynomial_decay":
                pruning_params = {
                    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.0,
                        final_sparsity=target_sparsity,
                        begin_step=0,
                        end_step=1000  # Will be adjusted based on epochs later
                    )
                }
            elif pruning_schedule == "constant_sparsity":
                pruning_params = {
                    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                        target_sparsity=target_sparsity,
                        begin_step=0
                    )
                }
            else:
                raise ValueError(f"Unsupported pruning schedule: {pruning_schedule}")
            
            # Apply pruning to the model
            pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
            
            return pruned_model
            
        except (ImportError, ModuleNotFoundError):
            # If TensorFlow Model Optimization is not installed, raise an error
            raise ImportError(
                "TensorFlow Model Optimization is required for pruning. "
                "Install it with: pip install tensorflow-model-optimization"
            )
    
    def get_pruning_callbacks(
        self, 
        update_freq: int = 100,
        log_dir: Optional[str] = None
    ) -> List[tf.keras.callbacks.Callback]:
        """Get callbacks needed for pruning.
        
        Args:
            update_freq: Frequency of weight updates
            log_dir: Directory to save pruning logs
            
        Returns:
            List of pruning callbacks
            
        Raises:
            ImportError: If TensorFlow Model Optimization is not installed
        """
        try:
            import tensorflow_model_optimization as tfmot
            
            callbacks = [
                tfmot.sparsity.keras.UpdatePruningStep(),
                tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir),
            ]
            
            return callbacks
            
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "TensorFlow Model Optimization is required for pruning. "
                "Install it with: pip install tensorflow-model-optimization"
            )
    
    def strip_pruning(self, model: tf.keras.Model) -> tf.keras.Model:
        """Remove pruning wrappers from the model for deployment.
        
        Args:
            model: Pruned Keras model
            
        Returns:
            Model with pruning configuration removed (but weights still pruned)
            
        Raises:
            ImportError: If TensorFlow Model Optimization is not installed
        """
        try:
            import tensorflow_model_optimization as tfmot
            
            # Strip the pruning wrappers
            stripped_model = tfmot.sparsity.keras.strip_pruning(model)
            
            return stripped_model
            
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                "TensorFlow Model Optimization is required for pruning. "
                "Install it with: pip install tensorflow-model-optimization"
            )
    
    def create_representative_dataset(
        self, 
        dataset: tf.data.Dataset, 
        num_samples: int = 100
    ) -> Callable:
        """Create a representative dataset function for quantization.
        
        Args:
            dataset: TensorFlow dataset to sample from
            num_samples: Number of samples to use
            
        Returns:
            Function that yields representative samples
        """
        def representative_dataset_gen():
            for i, (data, _) in enumerate(dataset):
                if i >= num_samples:
                    break
                yield [data]
        
        return representative_dataset_gen