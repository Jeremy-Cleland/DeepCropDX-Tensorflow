import tensorflow as tf
import os
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm.auto import tqdm
import glob
import math

from src.config.config import get_paths
from src.preprocessing.data_validator import DataValidator
from src.utils.seed_utils import set_global_seeds


def apply_perspective_transform(image, max_delta=0.1):
    """Apply a random perspective transformation to an image

    Args:
        image: A tensor of shape [height, width, channels]
        max_delta: Maximum distortion parameter

    Returns:
        Transformed image tensor of the same shape
    """
    height, width = tf.shape(image)[0], tf.shape(image)[1]

    # Create source points (4 corners of the image)
    src_points = tf.constant(
        [
            [0, 0],  # Top-left
            [width - 1, 0],  # Top-right
            [width - 1, height - 1],  # Bottom-right
            [0, height - 1],  # Bottom-left
        ],
        dtype=tf.float32,
    )

    # Create random offsets for destination points
    x_delta = tf.random.uniform(
        [4],
        -max_delta * tf.cast(width, tf.float32),
        max_delta * tf.cast(width, tf.float32),
    )
    y_delta = tf.random.uniform(
        [4],
        -max_delta * tf.cast(height, tf.float32),
        max_delta * tf.cast(height, tf.float32),
    )

    # Create destination points by adding offsets to source points
    dst_points = src_points + tf.stack([x_delta, y_delta], axis=1)

    # Convert to format expected by transform_projective
    src_points = tf.expand_dims(src_points, axis=0)
    dst_points = tf.expand_dims(dst_points, axis=0)

    # Create the homography matrix
    transform = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(image, 0),
        transforms=tf.raw_ops.ComputeProjectiveTransform(
            src=src_points, dst=dst_points
        ),
        output_shape=tf.shape(image)[0:2],
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )

    return tf.squeeze(transform, axis=0)


def random_erasing(image, p=0.5, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0):
    """Randomly erase rectangles in the image (occlusion)

    Args:
        image: A tensor of shape [height, width, channels]
        p: Probability of applying random erasing
        scale: Range of area proportion to erase
        ratio: Range of aspect ratio for erasing region
        value: Value to fill erased region (0 for black)

    Returns:
        Augmented image tensor
    """
    if tf.random.uniform(shape=(), minval=0, maxval=1) > p:
        return image

    height, width, channels = tf.shape(image)[0], tf.shape(image)[1], tf.shape(image)[2]
    area = tf.cast(height * width, tf.float32)

    # Choose random scale and ratio
    scale_factor = tf.random.uniform(shape=(), minval=scale[0], maxval=scale[1])
    target_area = area * scale_factor
    aspect_ratio = tf.random.uniform(shape=(), minval=ratio[0], maxval=ratio[1])

    # Calculate h and w of erasing rectangle
    h = tf.sqrt(target_area * aspect_ratio)
    w = tf.sqrt(target_area / aspect_ratio)
    h = tf.minimum(tf.cast(h, tf.int32), height)
    w = tf.minimum(tf.cast(w, tf.int32), width)

    # Choose random position
    i = tf.random.uniform(shape=(), minval=0, maxval=height - h + 1, dtype=tf.int32)
    j = tf.random.uniform(shape=(), minval=0, maxval=width - w + 1, dtype=tf.int32)

    # Create erasing mask
    mask = tf.ones_like(image)
    erasing_region = tf.zeros([h, w, channels])

    # Update mask with erasing region
    mask_indices = tf.stack([i, j], axis=0)
    mask_updates = tf.zeros([h, w, channels])

    # Create mask patches
    rows = tf.range(i, i + h)
    cols = tf.range(j, j + w)
    indices = tf.meshgrid(rows, cols)
    indices = tf.stack(indices, axis=-1)
    indices = tf.reshape(indices, [-1, 2])

    # Create the mask using scatter_nd
    mask_shape = tf.shape(image)
    mask = tf.ones(mask_shape, dtype=image.dtype)
    updates = tf.zeros([h * w, channels], dtype=image.dtype)
    mask = tf.tensor_scatter_nd_update(mask, indices, updates)

    # Apply mask to image
    erased_image = image * mask

    return erased_image


def add_gaussian_noise(image, mean=0.0, stddev=0.01):
    """Add Gaussian noise to an image

    Args:
        image: A tensor of shape [height, width, channels]
        mean: Mean of the Gaussian noise distribution
        stddev: Standard deviation of the noise

    Returns:
        Noisy image tensor
    """
    noise = tf.random.normal(shape=tf.shape(image), mean=mean, stddev=stddev)
    noisy_image = image + noise
    return tf.clip_by_value(noisy_image, 0.0, 1.0)


def apply_mixup(images, labels, alpha=0.2):
    """Apply MixUp augmentation to a batch of images and labels

    Args:
        images: Batch of images [batch_size, height, width, channels]
        labels: Batch of one-hot encoded labels [batch_size, num_classes]
        alpha: Beta distribution parameter

    Returns:
        Tuple of (mixed_images, mixed_labels)
    """
    batch_size = tf.shape(images)[0]

    # Create shuffled indices
    indices = tf.random.shuffle(tf.range(batch_size))

    # Sample mixing parameter from beta distribution
    lam = tf.random.uniform(shape=[batch_size], minval=0, maxval=1)
    if alpha > 0:
        lam = tf.random.beta(alpha, alpha, shape=[batch_size])

    # Ensure lambda is between 0 and 1
    lam_x = tf.maximum(lam, 1 - lam)
    lam_x = tf.reshape(lam_x, [-1, 1, 1, 1])

    # Mix images
    mixed_images = lam_x * images + (1 - lam_x) * tf.gather(images, indices)

    # Mix labels - reshape lambda for labels
    lam_y = tf.reshape(lam, [-1, 1])
    mixed_labels = lam_y * labels + (1 - lam_y) * tf.gather(labels, indices)

    return mixed_images, mixed_labels


def apply_cutmix(images, labels, alpha=1.0):
    """Apply CutMix augmentation to a batch of images and labels

    Args:
        images: Batch of images [batch_size, height, width, channels]
        labels: Batch of one-hot encoded labels [batch_size, num_classes]
        alpha: Beta distribution parameter

    Returns:
        Tuple of (mixed_images, mixed_labels)
    """
    batch_size = tf.shape(images)[0]
    image_height, image_width = tf.shape(images)[1], tf.shape(images)[2]

    # Create shuffled indices
    indices = tf.random.shuffle(tf.range(batch_size))

    # Sample mixing parameter from beta distribution
    lam = tf.random.beta(alpha, alpha, shape=[])

    # Sample rectangular box coordinates
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(image_height, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(image_width, tf.float32) * cut_ratio, tf.int32)

    # Ensure the box isn't empty
    cut_h = tf.maximum(cut_h, 1)
    cut_w = tf.maximum(cut_w, 1)

    # Generate random box center
    center_x = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)
    center_y = tf.random.uniform(
        shape=[], minval=0, maxval=image_height, dtype=tf.int32
    )

    # Calculate box boundaries
    box_x1 = tf.maximum(center_x - cut_w // 2, 0)
    box_y1 = tf.maximum(center_y - cut_h // 2, 0)
    box_x2 = tf.minimum(center_x + cut_w // 2, image_width)
    box_y2 = tf.minimum(center_y + cut_h // 2, image_height)

    # Create mask for the box
    outside_box = tf.logical_or(
        tf.logical_or(
            tf.less(tf.range(image_height)[:, tf.newaxis], box_y1),
            tf.greater(tf.range(image_height)[:, tf.newaxis], box_y2),
        )[:, tf.newaxis, :, tf.newaxis],
        tf.logical_or(
            tf.less(tf.range(image_width)[tf.newaxis, :], box_x1),
            tf.greater(tf.range(image_width)[tf.newaxis, :], box_x2),
        )[tf.newaxis, :, tf.newaxis, tf.newaxis],
    )

    # Expand mask to batch dimension
    mask = tf.cast(outside_box, images.dtype)

    # Calculate real lambda
    box_area = tf.cast((box_y2 - box_y1) * (box_x2 - box_x1), tf.float32)
    image_area = tf.cast(image_height * image_width, tf.float32)
    lam = 1.0 - (box_area / image_area)

    # Apply CutMix - first create copies of the original batch
    images_mixed = tf.identity(images)

    # Cut and paste the box from random images
    cut_indices = tf.range(batch_size)
    shuffled_indices = tf.gather(indices, cut_indices)

    # Mix the images
    images_mixed = images_mixed * mask + tf.gather(images, shuffled_indices) * (
        1 - mask
    )

    # Mix the labels
    lam = tf.cast(lam, labels.dtype)
    labels_mixed = lam * labels + (1 - lam) * tf.gather(labels, shuffled_indices)

    return images_mixed, labels_mixed


def enhanced_augmentation_pipeline(image, label, config=None):
    """Enhanced augmentation pipeline with multiple techniques

    Args:
        image: Input image tensor [height, width, channels]
        label: Input label tensor
        config: Dictionary of augmentation configuration parameters

    Returns:
        Tuple of (augmented_image, label)
    """
    if config is None:
        config = {}

    # Get augmentation parameters from config or use defaults
    apply_color_jitter = config.get("color_jitter", True)
    apply_noise = config.get("gaussian_noise", True)
    noise_stddev = config.get("noise_stddev", 0.01)
    apply_erasing = config.get("random_erasing", True)
    erasing_prob = config.get("erasing_prob", 0.1)
    apply_perspective = config.get("perspective_transform", True)
    perspective_delta = config.get("perspective_delta", 0.1)

    # Standard augmentations
    rotation_range = config.get("rotation_range", 20)
    width_shift_range = config.get("width_shift_range", 0.2)
    height_shift_range = config.get("height_shift_range", 0.2)
    horizontal_flip = config.get("horizontal_flip", True)
    vertical_flip = config.get("vertical_flip", False)

    # Random rotation
    if rotation_range > 0:
        radian = rotation_range * math.pi / 180
        angle = tf.random.uniform(
            shape=[],
            minval=-radian,
            maxval=radian,
        )
        image = tf.image.rot90(image, k=tf.cast(angle / (math.pi / 2), tf.int32))

    # Random translation
    if width_shift_range > 0 or height_shift_range > 0:
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        if width_shift_range > 0:
            w_pixels = tf.cast(image_width * width_shift_range, tf.int32)
            w_shift = tf.random.uniform(
                shape=[], minval=-w_pixels, maxval=w_pixels, dtype=tf.int32
            )
            image = tf.roll(image, shift=w_shift, axis=1)

        if height_shift_range > 0:
            h_pixels = tf.cast(image_height * height_shift_range, tf.int32)
            h_shift = tf.random.uniform(
                shape=[], minval=-h_pixels, maxval=h_pixels, dtype=tf.int32
            )
            image = tf.roll(image, shift=h_shift, axis=0)

    # Random flips
    if horizontal_flip and tf.random.uniform(shape=[]) > 0.5:
        image = tf.image.flip_left_right(image)

    if vertical_flip and tf.random.uniform(shape=[]) > 0.5:
        image = tf.image.flip_up_down(image)

    # Advanced augmentations

    # Color jitter
    if apply_color_jitter:
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.2)

        # Random contrast
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

        # Random saturation
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

        # Random hue
        image = tf.image.random_hue(image, max_delta=0.1)

    # Perspective transformation
    if apply_perspective and tf.random.uniform(shape=[]) > 0.5:
        image = apply_perspective_transform(image, max_delta=perspective_delta)

    # Gaussian noise
    if apply_noise and tf.random.uniform(shape=[]) > 0.5:
        image = add_gaussian_noise(image, stddev=noise_stddev)

    # Random erasing
    if apply_erasing and tf.random.uniform(shape=[]) < erasing_prob:
        image = random_erasing(
            image, p=1.0
        )  # p=1.0 because we already checked probability

    # Ensure image values stay in valid range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


def enhanced_batch_augmentation_pipeline(images, labels, config=None):
    """Apply batch-level augmentations like MixUp and CutMix

    Args:
        images: Batch of images [batch_size, height, width, channels]
        labels: Batch of one-hot encoded labels [batch_size, num_classes]
        config: Dictionary of augmentation configuration parameters

    Returns:
        Tuple of (augmented_images, augmented_labels)
    """
    if config is None:
        config = {}

    # Get augmentation parameters from config or use defaults
    apply_mixup = config.get("mixup", True)
    apply_cutmix = config.get("cutmix", True)
    mixup_alpha = config.get("mixup_alpha", 0.2)
    cutmix_alpha = config.get("cutmix_alpha", 1.0)

    # Select one batch augmentation randomly
    aug_choice = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)

    # Apply MixUp
    if aug_choice == 1 and apply_mixup:
        images, labels = apply_mixup(images, labels, alpha=mixup_alpha)

    # Apply CutMix
    elif aug_choice == 2 and apply_cutmix:
        images, labels = apply_cutmix(images, labels, alpha=cutmix_alpha)

    # Otherwise, no batch augmentation (orig_image, orig_label)

    return images, labels


def get_validation_transforms(image, label, image_size=(224, 224)):
    """Transforms for validation - center crop and normalization only

    Args:
        image: Input image tensor
        label: Input label tensor
        image_size: Target size (height, width)

    Returns:
        Tuple of (processed_image, label)
    """
    # Resize to slightly larger than target size
    larger_size = (int(image_size[0] * 1.14), int(image_size[1] * 1.14))
    image = tf.image.resize(image, larger_size)

    # Center crop to target size
    image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])

    # Ensure normalization
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label


class DataLoader:
    """
    Enhanced data loader using tf.data API for better performance and memory efficiency.
    Now with support for reproducible dataset splits.
    """

    def __init__(self, config):
        """Initialize the data loader with configuration

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = get_paths()

        # Set random seed if specified
        self.seed = self.config.get("seed", 42)
        set_global_seeds(self.seed)

        # Get training configuration
        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 32)
        self.validation_split = training_config.get("validation_split", 0.2)
        self.test_split = training_config.get("test_split", 0.1)

        # Get hardware configuration
        hardware_config = config.get("hardware", {})
        self.num_parallel_calls = hardware_config.get(
            "num_parallel_calls", tf.data.AUTOTUNE
        )
        self.prefetch_size = hardware_config.get(
            "prefetch_buffer_size", tf.data.AUTOTUNE
        )

        # Get data augmentation configuration
        self.augmentation_config = config.get("data_augmentation", {})

        # Get advanced augmentation configuration
        self.advanced_augmentation = config.get("advanced_augmentation", {})
        self.use_enhanced_augmentation = self.advanced_augmentation.get(
            "enabled", False
        )
        self.use_batch_augmentation = self.advanced_augmentation.get(
            "batch_augmentation", False
        )

        # Initialize data validator
        self.validator = DataValidator(config)

        # Get validation configuration
        validation_config = config.get("data_validation", {})
        self.validate_data = validation_config.get("enabled", True)

        # Set image parameters
        self.image_size = self.config.get("data", {}).get("image_size", (224, 224))
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)

    def load_data_efficient(self, data_dir=None, use_saved_splits=None):
        """Load data using efficient tf.data pipeline with sharding support

        Args:
            data_dir: Path to the dataset directory. If None, uses the configured path.
            use_saved_splits: Whether to try loading from saved splits first.
                              If None, determined from config.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset, class_names)
        """
        # Determine if we should use saved splits
        if use_saved_splits is None:
            use_saved_splits = self.config.get("data", {}).get(
                "use_saved_splits", False
            )

        if data_dir is None:
            # Use configured paths
            data_path_config = self.config.get("paths", {}).get("data", {})
            if isinstance(data_path_config, dict):
                data_dir = data_path_config.get("processed", "data/processed")
            else:
                data_dir = "data/processed"

        # Ensure the path is absolute
        data_dir = Path(data_dir)
        if not data_dir.is_absolute():
            data_dir = self.paths.base_dir / data_dir

        print(f"Loading data from {data_dir}")

        # Check if saved splits exist and should be used
        splits_dir = data_dir / "splits"
        splits_metadata_path = splits_dir / "splits_metadata.json"

        if use_saved_splits and splits_dir.exists() and splits_metadata_path.exists():
            try:
                print(f"Found saved splits at {splits_dir}, loading...")
                return self.load_from_saved_splits(splits_dir)
            except Exception as e:
                print(f"Failed to load from saved splits: {e}")
                print("Falling back to creating new splits")

        # Validate the dataset if enabled
        if self.validate_data:
            print("Validating dataset before loading...")
            validation_results = self.validator.validate_dataset(data_dir)

            # Check for critical errors that would prevent proper training
            if validation_results["errors"]:
                raise ValueError(
                    f"Dataset validation found critical errors: {validation_results['errors']}. "
                    "Please fix these issues before training."
                )

            # Log warnings but continue
            if validation_results["warnings"]:
                print("\nDataset validation warnings:")
                for warning in validation_results["warnings"]:
                    print(f"  - {warning}")
                print("\nContinuing with data loading despite warnings...\n")

        # Gather class directories and create label mapping
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
        if not class_dirs:
            raise ValueError(f"No class directories found in {data_dir}")

        # Sort class directories for reproducibility
        class_dirs = sorted(class_dirs)

        # Create class mapping
        class_names = {i: class_dir.name for i, class_dir in enumerate(class_dirs)}
        class_indices = {class_dir.name: i for i, class_dir in enumerate(class_dirs)}

        # Initialize lists for file paths and labels
        all_files = []
        all_labels = []
        class_counts = {}

        # Collect all files and labels
        print("Scanning dataset...")
        for class_dir in tqdm(class_dirs, desc="Classes"):
            class_name = class_dir.name
            class_idx = class_indices[class_name]
            class_counts[class_name] = 0

            # Find all image files in this class directory
            image_files = []
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                image_files.extend(list(class_dir.glob(f"**/{ext}")))
                image_files.extend(list(class_dir.glob(f"**/{ext.upper()}")))

            # Add files and labels to lists
            for img_file in image_files:
                all_files.append(str(img_file))
                all_labels.append(class_idx)
                class_counts[class_name] += 1

        # Print dataset statistics
        print(f"Dataset scan completed:")
        print(f"  - Total images: {len(all_files)}")
        print(f"  - Classes: {len(class_names)}")
        for class_name, count in class_counts.items():
            print(f"    - {class_name}: {count} images")

        # Create a tf.data.Dataset from file paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((all_files, all_labels))

        # Shuffle the dataset with a fixed seed for reproducibility
        dataset = dataset.shuffle(
            buffer_size=min(len(all_files), 10000),
            seed=self.seed,
            reshuffle_each_iteration=True,
        )

        # Check if there's a dedicated test directory
        test_dir = self.paths.data_dir / "test"
        separate_test_set = False
        test_dataset = None

        if test_dir.exists() and self.test_split > 0:
            print("Found dedicated test directory. Loading test set...")
            test_files = []
            test_labels = []

            # Assume same class structure in test directory
            for class_name, class_idx in class_indices.items():
                class_test_dir = test_dir / class_name
                if class_test_dir.exists():
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
                        test_files.extend(
                            [
                                (str(f), class_idx)
                                for f in class_test_dir.glob(f"**/{ext}")
                            ]
                        )
                        test_files.extend(
                            [
                                (str(f), class_idx)
                                for f in class_test_dir.glob(f"**/{ext.upper()}")
                            ]
                        )

            if test_files:
                separate_test_set = True
                test_files_paths, test_labels = zip(*test_files)
                test_dataset = tf.data.Dataset.from_tensor_slices(
                    (test_files_paths, test_labels)
                )
                print(
                    f"  - Test: {len(test_files)} images from dedicated test directory"
                )

        # Split the dataset if no separate test set is available
        if not separate_test_set:
            # Calculate split sizes
            dataset_size = len(all_files)
            train_size = int(
                dataset_size * (1 - self.validation_split - self.test_split)
            )
            val_size = int(dataset_size * self.validation_split)
            test_size = dataset_size - train_size - val_size

            print(f"Splitting dataset:")
            print(
                f"  - Training: {train_size} images ({(train_size/dataset_size)*100:.1f}%)"
            )
            print(
                f"  - Validation: {val_size} images ({(val_size/dataset_size)*100:.1f}%)"
            )
            print(f"  - Test: {test_size} images ({(test_size/dataset_size)*100:.1f}%)")

            # Split the dataset
            train_dataset = dataset.take(train_size)
            temp_dataset = dataset.skip(train_size)
            val_dataset = temp_dataset.take(val_size)
            test_dataset = temp_dataset.skip(val_size)
        else:
            # If we have a separate test set, just split into train and validation
            dataset_size = len(all_files)
            train_size = int(dataset_size * (1 - self.validation_split))
            val_size = dataset_size - train_size

            print(f"Splitting dataset (with separate test set):")
            print(
                f"  - Training: {train_size} images ({(train_size/dataset_size)*100:.1f}%)"
            )
            print(
                f"  - Validation: {val_size} images ({(val_size/dataset_size)*100:.1f}%)"
            )

            # Split the dataset
            train_dataset = dataset.take(train_size)
            val_dataset = dataset.skip(train_size)

        # Create preprocessing and augmentation functions
        def parse_image(file_path, label):
            """Load and preprocess an image from a file path."""
            # Read the image file
            img = tf.io.read_file(file_path)

            # Decode the image
            # Try different decoders based on file extension
            file_path_lower = tf.strings.lower(file_path)
            is_png = tf.strings.regex_full_match(file_path_lower, ".*\.png")
            is_jpeg = tf.strings.regex_full_match(file_path_lower, ".*\.(jpg|jpeg)")

            if is_png:
                img = tf.image.decode_png(img, channels=3)
            elif is_jpeg:
                img = tf.image.decode_jpeg(img, channels=3)
            else:
                # Default to image decoder which handles various formats
                img = tf.image.decode_image(img, channels=3, expand_animations=False)

            # Resize image
            img = tf.image.resize(img, self.image_size)

            # Normalize pixel values
            img = tf.cast(img, tf.float32) / 255.0

            # One-hot encode the label
            label = tf.one_hot(label, depth=len(class_names))

            return img, label

        def augment_image(image, label):
            """Apply data augmentation to an image."""
            if self.use_enhanced_augmentation:
                # Use the enhanced augmentation pipeline with our config
                return enhanced_augmentation_pipeline(
                    image, label, self.augmentation_config
                )

            # Original augmentation logic as fallback
            # Extract augmentation parameters from config
            rotation_range = self.augmentation_config.get("rotation_range", 20)
            width_shift_range = self.augmentation_config.get("width_shift_range", 0.2)
            height_shift_range = self.augmentation_config.get("height_shift_range", 0.2)
            zoom_range = self.augmentation_config.get("zoom_range", 0.2)
            horizontal_flip = self.augmentation_config.get("horizontal_flip", True)
            vertical_flip = self.augmentation_config.get("vertical_flip", False)

            # Random rotation
            if rotation_range > 0:
                angle = tf.random.uniform(
                    shape=[],
                    minval=-rotation_range,
                    maxval=rotation_range,
                    seed=self.seed,
                )
                image = tf.image.rot90(image, k=tf.cast(angle / 90, tf.int32))

            # Random width shift
            if width_shift_range > 0:
                w_shift = (
                    tf.random.uniform(
                        shape=[],
                        minval=-width_shift_range,
                        maxval=width_shift_range,
                        seed=self.seed,
                    )
                    * self.image_size[1]
                )
                image = tf.roll(image, shift=tf.cast(w_shift, tf.int32), axis=1)

            # Random height shift
            if height_shift_range > 0:
                h_shift = (
                    tf.random.uniform(
                        shape=[],
                        minval=-height_shift_range,
                        maxval=height_shift_range,
                        seed=self.seed,
                    )
                    * self.image_size[0]
                )
                image = tf.roll(image, shift=tf.cast(h_shift, tf.int32), axis=0)

            # Random horizontal flip
            if horizontal_flip:
                image = tf.image.random_flip_left_right(image, seed=self.seed)

            # Random vertical flip
            if vertical_flip:
                image = tf.image.random_flip_up_down(image, seed=self.seed)

            # Random brightness
            image = tf.image.random_brightness(image, 0.2, seed=self.seed)

            # Random contrast
            image = tf.image.random_contrast(image, 0.8, 1.2, seed=self.seed)

            # Make sure pixel values are still in [0, 1]
            image = tf.clip_by_value(image, 0.0, 1.0)

            return image, label

        # Define a function for batch augmentation
        def apply_batch_augmentation(images, labels):
            """Apply batch-level augmentation (MixUp, CutMix) to a batch"""
            if not self.use_batch_augmentation:
                return images, labels

            return enhanced_batch_augmentation_pipeline(
                images, labels, self.advanced_augmentation
            )

        # Apply preprocessing to datasets
        print("Preparing datasets...")

        # Apply image loading and preprocessing to all datasets
        train_dataset = train_dataset.map(
            parse_image, num_parallel_calls=self.num_parallel_calls
        )
        val_dataset = val_dataset.map(
            parse_image, num_parallel_calls=self.num_parallel_calls
        )
        if test_dataset is not None:
            test_dataset = test_dataset.map(
                parse_image, num_parallel_calls=self.num_parallel_calls
            )

        # Apply augmentation only to training data
        if self.augmentation_config.get("enabled", True):
            print("Applying data augmentation to training set")
            train_dataset = train_dataset.map(
                augment_image, num_parallel_calls=self.num_parallel_calls
            )

        # Batch datasets
        train_dataset = train_dataset.batch(self.batch_size)
        val_dataset = val_dataset.batch(self.batch_size)
        if test_dataset is not None:
            test_dataset = test_dataset.batch(self.batch_size)

        # Apply batch augmentation to the training dataset if enabled
        if self.use_batch_augmentation:
            print("Applying batch augmentation (MixUp/CutMix) to training set")
            train_dataset = train_dataset.map(
                apply_batch_augmentation, num_parallel_calls=self.num_parallel_calls
            )

        # Apply prefetch for all datasets
        train_dataset = train_dataset.prefetch(self.prefetch_size)
        val_dataset = val_dataset.prefetch(self.prefetch_size)
        if test_dataset is not None:
            test_dataset = test_dataset.prefetch(self.prefetch_size)

        # Add properties to make compatible with Keras generators
        # This helps maintain compatibility with existing code
        class_indices_dict = {name: idx for idx, name in class_names.items()}

        # Create generator-like attributes for train dataset
        train_dataset.class_indices = class_indices_dict
        train_dataset.samples = train_size

        # Create generator-like attributes for validation dataset
        val_dataset.class_indices = class_indices_dict
        val_dataset.samples = val_size

        # Create generator-like attributes for test dataset if it exists
        if test_dataset is not None:
            test_dataset.class_indices = class_indices_dict
            test_dataset.samples = (
                test_size if not separate_test_set else len(test_files)
            )

        print(f"Dataset preparation complete.")

        # Save the splits for future use if enabled
        if self.config.get("data", {}).get("save_splits", False):
            try:
                print("Saving dataset splits for reproducibility")
                self.save_dataset_splits(
                    {"train": train_dataset, "val": val_dataset, "test": test_dataset},
                    class_indices,
                    data_dir,
                )
            except Exception as e:
                print(f"Failed to save dataset splits: {e}")

        return train_dataset, val_dataset, test_dataset, class_names

    def load_data(self, data_dir=None, use_saved_splits=None):
        """
        Load data with support for saved splits for reproducibility.

        Args:
            data_dir: Path to the dataset directory. If None, uses the configured path.
            use_saved_splits: Whether to try loading from saved splits first.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset, class_names)
        """
        # Determine if we should use saved splits
        if use_saved_splits is None:
            use_saved_splits = self.config.get("data", {}).get(
                "use_saved_splits", False
            )

        return self.load_data_efficient(data_dir, use_saved_splits)

    def save_dataset_splits(self, dataset_info, class_indices, output_dir):
        """Save dataset splits to disk for reproducibility

        Args:
            dataset_info: Dictionary containing dataset information with keys 'train', 'val', 'test'
            class_indices: Dictionary mapping class names to indices
            output_dir: Directory to save the splits

        Returns:
            Dictionary with paths to saved split files
        """
        output_path = Path(output_dir)
        splits_dir = output_path / "splits"
        os.makedirs(splits_dir, exist_ok=True)

        # Save class mapping
        class_to_idx = class_indices
        idx_to_class = {idx: name for name, idx in class_to_idx.items()}

        class_mapping_path = splits_dir / "class_mapping.json"
        with open(class_mapping_path, "w") as f:
            json.dump(
                {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class},
                f,
                indent=2,
            )

        print(f"Class mapping saved to {class_mapping_path}")

        split_paths = {}

        # Extract and save file paths and labels for each split
        for split_name, dataset in dataset_info.items():
            if dataset is None:
                continue

            # Create lists to store file paths, indices and labels
            file_paths = []
            indices = []
            labels = []

            # Try to get file paths directly from source before any transformations
            source_dataset = getattr(dataset, "_input_dataset", None)
            if source_dataset is not None:
                print(
                    f"Found source dataset for {split_name}, attempting to extract file paths"
                )

                # Try different approaches to get file paths
                try:
                    # For datasets created from tensor_slices((file_paths, labels))
                    for element in source_dataset.take(1):
                        if isinstance(element, tuple) and isinstance(
                            element[0], tf.Tensor
                        ):
                            if element[0].dtype == tf.string:
                                # This is likely a file path tensor
                                print(
                                    f"Found file path tensor in source dataset for {split_name}"
                                )

                                # Create a new dataset that extracts just the file paths and labels
                                for file_path_tensor, label_tensor in source_dataset:
                                    try:
                                        # Convert tensor to numpy and then to string
                                        file_path = file_path_tensor.numpy().decode(
                                            "utf-8"
                                        )
                                        label = label_tensor.numpy()

                                        file_paths.append(file_path)
                                        labels.append(label)
                                    except:
                                        pass
                except:
                    print(
                        f"Could not extract file paths from source dataset for {split_name}"
                    )

            # If we got file paths directly from source
            if file_paths:
                print(
                    f"Successfully extracted {len(file_paths)} file paths for {split_name}"
                )

                # Create DataFrame with file paths
                split_df = pd.DataFrame(
                    {
                        "file_path": file_paths,
                        "label": labels,
                        "class_name": [
                            idx_to_class.get(
                                str(label) if isinstance(label, str) else label,
                                f"unknown_{label}",
                            )
                            for label in labels
                        ],
                    }
                )
            else:
                print(
                    f"Could not extract file paths for {split_name} split, will save index only"
                )

                # Extract indices and labels
                index_counter = 0

                # Get number of samples if possible
                num_samples = getattr(dataset, "samples", None)

                # Process each batch
                for x_batch, y_batch in tqdm(
                    dataset, desc=f"Processing {split_name} split", total=num_samples
                ):
                    # Handle batched data
                    batch_size = x_batch.shape[0]

                    # Extract labels from the batch
                    if (
                        len(y_batch.shape) > 1 and y_batch.shape[1] > 1
                    ):  # One-hot encoded
                        batch_labels = tf.argmax(y_batch, axis=1).numpy()
                    else:
                        batch_labels = y_batch.numpy()

                    # Add indices and labels
                    for i in range(batch_size):
                        indices.append(index_counter)
                        labels.append(int(batch_labels[i]))
                        index_counter += 1

                # Create DataFrame without file paths
                split_df = pd.DataFrame(
                    {
                        "index": indices,
                        "label": labels,
                        "class_name": [
                            idx_to_class.get(label, f"unknown_{label}")
                            for label in labels
                        ],
                    }
                )

            # Save to CSV
            split_path = splits_dir / f"{split_name}_split.csv"
            split_df.to_csv(split_path, index=False)
            split_paths[split_name] = str(split_path)

            print(
                f"{split_name.capitalize()} split saved to {split_path} ({len(split_df)} samples)"
            )

        # Save metadata about the splits
        metadata = {
            "splits": {
                k: {"path": v, "size": len(pd.read_csv(v))}
                for k, v in split_paths.items()
            },
            "class_mapping_path": str(class_mapping_path),
            "num_classes": len(class_to_idx),
            "image_size": list(self.image_size),
            "creation_timestamp": str(pd.Timestamp.now()),
            "split_percentages": {
                "validation_split": self.validation_split,
                "test_split": self.test_split,
            },
            "has_file_paths": {
                k: "file_path" in pd.read_csv(v).columns for k, v in split_paths.items()
            },
        }

        metadata_path = splits_dir / "splits_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset splits metadata saved to {metadata_path}")
        return split_paths

    def load_from_saved_splits(self, splits_dir=None):
        """Load datasets from previously saved splits

        Args:
            splits_dir: Directory containing the saved splits. If None, uses the default processed data directory.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset, class_names)
        """
        if splits_dir is None:
            # Use configured paths
            data_path_config = self.config.get("paths", {}).get("data", {})
            if isinstance(data_path_config, dict):
                data_dir = data_path_config.get("processed", "data/processed")
            else:
                data_dir = "data/processed"

            # Ensure the path is absolute
            data_dir = Path(data_dir)
            if not data_dir.is_absolute():
                data_dir = self.paths.base_dir / data_dir

            splits_dir = data_dir / "splits"
        else:
            splits_dir = Path(splits_dir)

        if not splits_dir.exists():
            raise ValueError(f"Splits directory not found at {splits_dir}")

        # Load metadata if available
        metadata_path = splits_dir / "splits_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            print(f"Loaded splits metadata from {metadata_path}")
            print(
                f"Found {metadata['num_classes']} classes with splits: {list(metadata['splits'].keys())}"
            )
        else:
            metadata = None
            print(f"No splits metadata found at {metadata_path}")

        # Load class mapping
        class_mapping_path = splits_dir / "class_mapping.json"
        if not class_mapping_path.exists():
            raise ValueError(f"Class mapping file not found at {class_mapping_path}")

        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)

        class_to_idx = class_mapping.get("class_to_idx", {})
        idx_to_class = class_mapping.get("idx_to_class", {})

        # Convert string indices to integers if needed
        if all(isinstance(k, str) for k in idx_to_class.keys()):
            idx_to_class = {int(k): v for k, v in idx_to_class.items()}

        class_names = {int(idx): name for idx, name in idx_to_class.items()}

        print(f"Loaded class mapping with {len(class_names)} classes")

        # Function to create a dataset from a CSV file
        def create_dataset_from_csv(csv_path):
            """Create a TensorFlow dataset from the CSV file containing split information"""
            if not Path(csv_path).exists():
                print(f"Split file not found at {csv_path}, skipping")
                return None

            # Load the CSV file
            df = pd.read_csv(csv_path)
            print(f"Loaded split with {len(df)} samples")

            # Check if we have file paths
            has_file_paths = "file_path" in df.columns

            if has_file_paths:
                # Create dataset from file paths and labels
                file_paths = df["file_path"].values
                labels = df["label"].values

                # Verify file paths exist
                valid_paths = []
                valid_labels = []
                for i, path in enumerate(file_paths):
                    if os.path.exists(path):
                        valid_paths.append(path)
                        valid_labels.append(labels[i])

                if len(valid_paths) < len(file_paths):
                    print(
                        f"Warning: {len(file_paths) - len(valid_paths)} file paths don't exist"
                    )
                    if len(valid_paths) == 0:
                        print(
                            "No valid file paths found, falling back to rebuilding dataset"
                        )
                        return None

                # Create a dataset from file paths and labels
                dataset = tf.data.Dataset.from_tensor_slices(
                    (valid_paths, valid_labels)
                )

                # Map the parse_image function
                dataset = dataset.map(
                    self._parse_image_with_class_names(class_names),
                    num_parallel_calls=self.num_parallel_calls,
                )
                return dataset
            else:
                # Don't create placeholder images - it would crash with large datasets
                print(
                    f"Original file paths not available for {os.path.basename(csv_path)}."
                )
                print("Falling back to rebuilding dataset from raw files.")
                return None

        # Load datasets from CSV files
        train_split_path = splits_dir / "train_split.csv"
        val_split_path = splits_dir / "val_split.csv"
        test_split_path = splits_dir / "test_split.csv"

        # Create datasets
        train_dataset = create_dataset_from_csv(train_split_path)
        val_dataset = create_dataset_from_csv(val_split_path)
        test_dataset = create_dataset_from_csv(test_split_path)

        # Check if required datasets were loaded - if not, rebuild from raw data
        if train_dataset is None or val_dataset is None:
            print("Could not load datasets from splits. Rebuilding from raw data...")
            # Get the parent directory of the splits directory
            parent_dir = splits_dir.parent

            # Fall back to creating new splits from the original data
            return self.load_data_efficient(parent_dir, use_saved_splits=False)

        # Apply augmentation to training set if enabled
        if self.augmentation_config.get("enabled", True):
            print("Applying data augmentation to training set")
            train_dataset = train_dataset.map(
                self._get_augment_function(), num_parallel_calls=self.num_parallel_calls
            )

        # Apply standard dataset transformations
        # Shuffle training data
        train_dataset = train_dataset.shuffle(
            buffer_size=10000, seed=self.seed, reshuffle_each_iteration=True
        )

        # Batch and prefetch
        train_dataset = train_dataset.batch(self.batch_size).prefetch(
            self.prefetch_size
        )
        val_dataset = val_dataset.batch(self.batch_size).prefetch(self.prefetch_size)
        if test_dataset is not None:
            test_dataset = test_dataset.batch(self.batch_size).prefetch(
                self.prefetch_size
            )

        # Add properties to make compatible with Keras generators
        class_indices_dict = {name: idx for name, idx in class_to_idx.items()}

        # Get sample counts from metadata if available, otherwise from DataFrames
        if metadata and "splits" in metadata:
            train_samples = (
                metadata["splits"]
                .get("train", {})
                .get("size", len(pd.read_csv(train_split_path)))
            )
            val_samples = (
                metadata["splits"]
                .get("val", {})
                .get("size", len(pd.read_csv(val_split_path)))
            )
            test_samples = metadata["splits"].get("test", {}).get("size", 0)
            if test_dataset is not None and test_samples == 0:
                test_samples = len(pd.read_csv(test_split_path))
        else:
            train_samples = len(pd.read_csv(train_split_path))
            val_samples = len(pd.read_csv(val_split_path))
            test_samples = 0
            if test_dataset is not None:
                test_samples = len(pd.read_csv(test_split_path))

        # Set dataset attributes
        train_dataset.class_indices = class_indices_dict
        train_dataset.samples = train_samples

        val_dataset.class_indices = class_indices_dict
        val_dataset.samples = val_samples

        if test_dataset is not None:
            test_dataset.class_indices = class_indices_dict
            test_dataset.samples = test_samples

        print(f"Successfully loaded datasets from saved splits at {splits_dir}")
        print(
            f"Train: {train_samples} samples, Validation: {val_samples} samples, Test: {test_samples if test_dataset is not None else 0} samples"
        )

        return train_dataset, val_dataset, test_dataset, class_names

    def _parse_image_with_class_names(self, class_names):
        """Return a function that parses images with the given class names"""

        def parse_image(file_path, label):
            """Load and preprocess an image from a file path."""
            # Read the image file
            img = tf.io.read_file(file_path)

            # Decode the image
            # Try different decoders based on file extension
            file_path_lower = tf.strings.lower(file_path)
            is_png = tf.strings.regex_full_match(file_path_lower, ".*\.png")
            is_jpeg = tf.strings.regex_full_match(file_path_lower, ".*\.(jpg|jpeg)")

            if is_png:
                img = tf.image.decode_png(img, channels=3)
            elif is_jpeg:
                img = tf.image.decode_jpeg(img, channels=3)
            else:
                # Default to image decoder which handles various formats
                img = tf.image.decode_image(img, channels=3, expand_animations=False)

            # Resize image
            img = tf.image.resize(img, self.image_size)

            # Normalize pixel values
            img = tf.cast(img, tf.float32) / 255.0

            # One-hot encode the label
            label = tf.one_hot(label, depth=len(class_names))

            return img, label

        return parse_image

    def _get_augment_function(self):
        """Return the data augmentation function with current configuration"""

        def augment_image(image, label):
            """Apply data augmentation to an image."""
            if self.use_enhanced_augmentation:
                # Use the enhanced augmentation pipeline with our config
                return enhanced_augmentation_pipeline(
                    image, label, self.augmentation_config
                )

            # Original augmentation logic as fallback
            # Extract augmentation parameters from config
            rotation_range = self.augmentation_config.get("rotation_range", 20)
            width_shift_range = self.augmentation_config.get("width_shift_range", 0.2)
            height_shift_range = self.augmentation_config.get("height_shift_range", 0.2)
            zoom_range = self.augmentation_config.get("zoom_range", 0.2)
            horizontal_flip = self.augmentation_config.get("horizontal_flip", True)
            vertical_flip = self.augmentation_config.get("vertical_flip", False)

            # Random rotation
            if rotation_range > 0:
                angle = tf.random.uniform(
                    shape=[],
                    minval=-rotation_range,
                    maxval=rotation_range,
                    seed=self.seed,
                )
                image = tf.image.rot90(image, k=tf.cast(angle / 90, tf.int32))

            # Random width shift
            if width_shift_range > 0:
                w_shift = (
                    tf.random.uniform(
                        shape=[],
                        minval=-width_shift_range,
                        maxval=width_shift_range,
                        seed=self.seed,
                    )
                    * self.image_size[1]
                )
                image = tf.roll(image, shift=tf.cast(w_shift, tf.int32), axis=1)

            # Random height shift
            if height_shift_range > 0:
                h_shift = (
                    tf.random.uniform(
                        shape=[],
                        minval=-height_shift_range,
                        maxval=height_shift_range,
                        seed=self.seed,
                    )
                    * self.image_size[0]
                )
                image = tf.roll(image, shift=tf.cast(h_shift, tf.int32), axis=0)

            # Random horizontal flip
            if horizontal_flip:
                image = tf.image.random_flip_left_right(image, seed=self.seed)

            # Random vertical flip
            if vertical_flip:
                image = tf.image.random_flip_up_down(image, seed=self.seed)

            # Random brightness
            image = tf.image.random_brightness(image, 0.2, seed=self.seed)

            # Random contrast
            image = tf.image.random_contrast(image, 0.8, 1.2, seed=self.seed)

            # Make sure pixel values are still in [0, 1]
            image = tf.clip_by_value(image, 0.0, 1.0)

            return image, label

        return augment_image
