"""
Dataset pipeline module for creating TensorFlow dataset pipelines.
This handles creating efficient pipelines from raw data files.
"""

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union, Callable

from src.preprocessing.data_transformations import (
    get_standard_augmentation_pipeline,
    get_enhanced_augmentation_pipeline,
    get_batch_augmentation_pipeline,
    get_validation_transforms,
    normalize_image,
    resize_image,
)


class DatasetPipeline:
    """Creates and manages TensorFlow dataset pipelines from file paths and labels."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset pipeline with given configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config

        # Set image parameters
        self.image_size = self.config.get("data", {}).get("image_size", (224, 224))
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)

        # Get batch size from training config
        training_config = self.config.get("training", {})
        self.batch_size = training_config.get("batch_size", 32)

        # Configure data pipeline parallelism
        hardware_config = self.config.get("hardware", {})
        self.num_parallel_calls = hardware_config.get(
            "num_parallel_calls", tf.data.AUTOTUNE
        )
        self.prefetch_size = hardware_config.get(
            "prefetch_buffer_size", tf.data.AUTOTUNE
        )

        # Get augmentation configuration
        self.augmentation_config = config.get("data_augmentation", {})
        self.use_enhanced_augmentation = self.augmentation_config.get("enabled", True)

        # Get advanced augmentation configuration
        self.advanced_augmentation = config.get("advanced_augmentation", {})
        self.use_batch_augmentation = self.advanced_augmentation.get(
            "batch_augmentation", False
        )

    def create_training_pipeline(
        self,
        file_paths: List[str],
        labels: List[int],
        indices: List[int],
        num_classes: int,
        shuffle_buffer: int = 10000,
    ) -> tf.data.Dataset:
        """Create a training dataset pipeline.

        Args:
            file_paths: List of file paths
            labels: List of labels
            indices: List of indices to use from file_paths and labels
            num_classes: Number of classes
            shuffle_buffer: Size of the shuffle buffer

        Returns:
            Training dataset pipeline
        """
        # Extract the specific file paths and labels using indices
        train_paths = [file_paths[i] for i in indices]
        train_labels = [labels[i] for i in indices]

        # Create a dataset from the file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

        # Shuffle the dataset
        dataset = dataset.shuffle(
            buffer_size=min(len(train_paths), shuffle_buffer),
            reshuffle_each_iteration=True,
        )

        # Parse images
        dataset = dataset.map(
            self._get_parse_function(num_classes),
            num_parallel_calls=self.num_parallel_calls,
        )

        # Apply augmentation
        if self.augmentation_config.get("enabled", True):
            if self.use_enhanced_augmentation:
                dataset = dataset.map(
                    get_enhanced_augmentation_pipeline(self.augmentation_config),
                    num_parallel_calls=self.num_parallel_calls,
                )
            else:
                dataset = dataset.map(
                    get_standard_augmentation_pipeline(self.augmentation_config),
                    num_parallel_calls=self.num_parallel_calls,
                )

        # Batch the dataset
        dataset = dataset.batch(self.batch_size)

        # Apply batch augmentation if enabled
        if self.use_batch_augmentation:
            dataset = dataset.map(
                get_batch_augmentation_pipeline(self.advanced_augmentation),
                num_parallel_calls=self.num_parallel_calls,
            )

        # Prefetch for better performance
        dataset = dataset.prefetch(self.prefetch_size)

        # Add dataset properties to make compatible with Keras
        class_indices = {i: str(i) for i in range(num_classes)}
        dataset.class_indices = class_indices
        dataset.samples = len(train_paths)

        return dataset

    def create_validation_pipeline(
        self,
        file_paths: List[str],
        labels: List[int],
        indices: List[int],
        num_classes: int,
    ) -> tf.data.Dataset:
        """Create a validation dataset pipeline.

        Args:
            file_paths: List of file paths
            labels: List of labels
            indices: List of indices to use from file_paths and labels
            num_classes: Number of classes

        Returns:
            Validation dataset pipeline
        """
        # Extract the specific file paths and labels using indices
        val_paths = [file_paths[i] for i in indices]
        val_labels = [labels[i] for i in indices]

        # Create a dataset from the file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))

        # Parse images
        dataset = dataset.map(
            self._get_parse_function(num_classes),
            num_parallel_calls=self.num_parallel_calls,
        )

        # Apply validation transforms
        dataset = dataset.map(
            get_validation_transforms(self.image_size),
            num_parallel_calls=self.num_parallel_calls,
        )

        # Batch the dataset
        dataset = dataset.batch(self.batch_size)

        # Prefetch for better performance
        dataset = dataset.prefetch(self.prefetch_size)

        # Add dataset properties to make compatible with Keras
        class_indices = {i: str(i) for i in range(num_classes)}
        dataset.class_indices = class_indices
        dataset.samples = len(val_paths)

        return dataset

    def create_test_pipeline(
        self,
        file_paths: List[str],
        labels: List[int],
        indices: List[int],
        num_classes: int,
    ) -> tf.data.Dataset:
        """Create a test dataset pipeline.

        Args:
            file_paths: List of file paths
            labels: List of labels
            indices: List of indices to use from file_paths and labels
            num_classes: Number of classes

        Returns:
            Test dataset pipeline
        """
        # This is essentially the same as validation pipeline
        return self.create_validation_pipeline(file_paths, labels, indices, num_classes)

    def _get_parse_function(self, num_classes: int) -> Callable:
        """Get a function that parses an image file and its label.

        Args:
            num_classes: Number of classes for one-hot encoding

        Returns:
            Function that takes (file_path, label) and returns (image, one_hot_label)
        """

        def parse_image_file(file_path, label):
            """Parse an image file and its label."""
            # Read the image file
            img = tf.io.read_file(file_path)

            # Decode the image
            # Try different decoders based on file extension
            file_path_lower = tf.strings.lower(file_path)
            is_png = tf.strings.regex_full_match(file_path_lower, ".*\\.png")
            is_jpeg = tf.strings.regex_full_match(file_path_lower, ".*\\.(jpg|jpeg)")

            def decode_png():
                return tf.image.decode_png(img, channels=3)

            def decode_jpeg():
                return tf.image.decode_jpeg(img, channels=3)

            def decode_other():
                return tf.image.decode_image(img, channels=3, expand_animations=False)

            # Use a conditional to choose the right decoder
            image = tf.cond(
                is_png, decode_png, lambda: tf.cond(is_jpeg, decode_jpeg, decode_other)
            )

            # Ensure the image has 3 channels
            image = tf.ensure_shape(image, [None, None, 3])

            # Resize image
            image = resize_image(image, self.image_size)

            # Normalize pixel values
            image = normalize_image(image, method="scale")

            # One-hot encode the label
            one_hot_label = tf.one_hot(label, depth=num_classes)

            return image, one_hot_label

        return parse_image_file

    @staticmethod
    def get_file_extension_tensors():
        """Get TensorFlow string patterns for common image file extensions.

        Returns:
            Dictionary of file extension pattern tensors
        """
        return {
            "png": tf.constant(".png"),
            "jpg": tf.constant(".jpg"),
            "jpeg": tf.constant(".jpeg"),
            "bmp": tf.constant(".bmp"),
        }
