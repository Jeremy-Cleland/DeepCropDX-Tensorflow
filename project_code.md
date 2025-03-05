# Project Code Overview

*Generated on 2025-03-05 02:56:17*

## Table of Contents

- [compile.py](#compile-py)
- [deepcropdx.yml](#deepcropdx-yml)
- [docs/data_loader_old.py](#docs-data_loader_old-py)
- [docs/model_factory_old.py](#docs-model_factory_old-py)
- [docs/test_data_loader.py](#docs-test_data_loader-py)
- [setup.py](#setup-py)
- [src/config/__init__.py](#src-config-__init__-py)
- [src/config/config.py](#src-config-config-py)
- [src/config/config.yaml](#src-config-config-yaml)
- [src/config/config_loader.py](#src-config-config_loader-py)
- [src/config/config_manager.py](#src-config-config_manager-py)
- [src/config/model_configs/__init__.py](#src-config-model_configs-__init__-py)
- [src/config/model_configs/models.yaml](#src-config-model_configs-models-yaml)
- [src/evaluation/__init__.py](#src-evaluation-__init__-py)
- [src/evaluation/metrics.py](#src-evaluation-metrics-py)
- [src/evaluation/visualization.py](#src-evaluation-visualization-py)
- [src/main.py](#src-main-py)
- [src/model_registry/__init__.py](#src-model_registry-__init__-py)
- [src/model_registry/registry_manager.py](#src-model_registry-registry_manager-py)
- [src/models/__init__.py](#src-models-__init__-py)
- [src/models/advanced_architectures.py](#src-models-advanced_architectures-py)
- [src/models/attention.py](#src-models-attention-py)
- [src/models/model_factory.py](#src-models-model_factory-py)
- [src/models/model_optimizer.py](#src-models-model_optimizer-py)
- [src/preprocessing/__init__.py](#src-preprocessing-__init__-py)
- [src/preprocessing/data_loader.py](#src-preprocessing-data_loader-py)
- [src/preprocessing/data_transformations.py](#src-preprocessing-data_transformations-py)
- [src/preprocessing/data_validator.py](#src-preprocessing-data_validator-py)
- [src/preprocessing/dataset_loader.py](#src-preprocessing-dataset_loader-py)
- [src/preprocessing/dataset_pipeline.py](#src-preprocessing-dataset_pipeline-py)
- [src/scripts/__init__.py](#src-scripts-__init__-py)
- [src/scripts/evaluate.py](#src-scripts-evaluate-py)
- [src/scripts/preprocess_data.py](#src-scripts-preprocess_data-py)
- [src/scripts/registry_cli.py](#src-scripts-registry_cli-py)
- [src/scripts/train.py](#src-scripts-train-py)
- [src/training/__init__.py](#src-training-__init__-py)
- [src/training/batch_trainer.py](#src-training-batch_trainer-py)
- [src/training/learning_rate_scheduler.py](#src-training-learning_rate_scheduler-py)
- [src/training/lr_finder.py](#src-training-lr_finder-py)
- [src/training/model_trainer.py](#src-training-model_trainer-py)
- [src/training/trainer.py](#src-training-trainer-py)
- [src/training/training_pipeline.py](#src-training-training_pipeline-py)
- [src/utils/__init__.py](#src-utils-__init__-py)
- [src/utils/cli_utils.py](#src-utils-cli_utils-py)
- [src/utils/error_handling.py](#src-utils-error_handling-py)
- [src/utils/hardware_utils.py](#src-utils-hardware_utils-py)
- [src/utils/logger.py](#src-utils-logger-py)
- [src/utils/memory_utils.py](#src-utils-memory_utils-py)
- [src/utils/report_generator.py](#src-utils-report_generator-py)
- [src/utils/seed_utils.py](#src-utils-seed_utils-py)

## Code Files

### compile.py

```python
#!/usr/bin/env python3
"""
Script to compile Python and YAML files in a project into a single markdown document.
This is useful for sharing code with LLMs for analysis.
"""

import os
import argparse
from datetime import datetime


def find_files(directory, extensions=[".py", ".yaml", ".yml"], ignore_dirs=None):
    """Find all files with specified extensions in the given directory and its subdirectories."""
    if ignore_dirs is None:
        ignore_dirs = [
            ".git",
            ".github",
            "__pycache__",
            "venv",
            "env",
            ".venv",
            ".env",
            "build",
            "dist",
        ]

    matching_files = []

    for root, dirs, files in os.walk(directory):
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]

        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                matching_files.append(os.path.join(root, file))

    return sorted(matching_files)


def get_relative_path(file_path, base_dir):
    """Get the path relative to the base directory."""
    return os.path.relpath(file_path, base_dir)


def get_file_language(file_path):
    """Determine the language based on file extension."""
    if file_path.endswith((".yaml", ".yml")):
        return "yaml"
    elif file_path.endswith(".py"):
        return "python"
    else:
        return "text"


def create_markdown(files, base_dir, output_file):
    """Create a markdown document from the files."""
    # Count file types
    python_files = [f for f in files if f.endswith(".py")]
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# Project Code Overview\n\n")
        f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        f.write("## Table of Contents\n\n")

        # Generate table of contents
        for file_path in files:
            rel_path = get_relative_path(file_path, base_dir)
            anchor = rel_path.replace("/", "-").replace(".", "-").replace(" ", "-")
            f.write(f"- [{rel_path}](#{anchor})\n")

        f.write("\n## Code Files\n\n")

        # Write each file with its content
        for file_path in files:
            rel_path = get_relative_path(file_path, base_dir)
            anchor = rel_path.replace("/", "-").replace(".", "-").replace(" ", "-")
            language = get_file_language(file_path)

            f.write(f"### {rel_path}\n\n")
            f.write(f"```{language}\n")

            try:
                with open(file_path, "r", encoding="utf-8") as code_file:
                    f.write(code_file.read())
            except UnicodeDecodeError:
                try:
                    # Try with a different encoding if UTF-8 fails
                    with open(file_path, "r", encoding="latin-1") as code_file:
                        f.write(code_file.read())
                except Exception as e:
                    f.write(f"# Error reading file: {str(e)}\n")
            except Exception as e:
                f.write(f"# Error reading file: {str(e)}\n")

            f.write("```\n\n")
            f.write("---\n\n")

        # Add summary information
        f.write(f"## Summary\n\n")
        f.write(f"Total files: {len(files)}\n")
        f.write(f"- Python files: {len(python_files)}\n")
        f.write(f"- YAML files: {len(yaml_files)}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compile Python and YAML files into a markdown document."
    )
    parser.add_argument(
        "--directory",
        "-d",
        type=str,
        default=".",
        help="Directory to scan for files (default: current directory)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="project_code.md",
        help="Output markdown file (default: project_code.md)",
    )
    parser.add_argument(
        "--ignore",
        "-i",
        type=str,
        nargs="+",
        default=[
            ".git",
            ".github",
            "__pycache__",
            "venv",
            "env",
            ".venv",
            ".env",
            "build",
            "dist",
        ],
        help="Directories to ignore (default: .git, __pycache__, venv, etc.)",
    )
    parser.add_argument(
        "--extensions",
        "-e",
        type=str,
        nargs="+",
        default=[".py", ".yaml", ".yml"],
        help="File extensions to include (default: .py, .yaml, .yml)",
    )

    args = parser.parse_args()

    base_dir = os.path.abspath(args.directory)
    files = find_files(base_dir, args.extensions, args.ignore)

    if not files:
        print(f"No matching files found in {base_dir}")
        return

    create_markdown(files, base_dir, args.output)
    print(f"Markdown document created at {args.output}")
    print(f"Found {len(files)} files")

    # Print breakdown by type
    python_files = [f for f in files if f.endswith(".py")]
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]
    print(f"- {len(python_files)} Python files")
    print(f"- {len(yaml_files)} YAML files")


if __name__ == "__main__":
    main()
```

---

### deepcropdx.yml

```yaml
name: deepcropdx
channels:
  - conda-forge
  - apple
  - defaults
dependencies:
  - python=3.10
  - pip
  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scipy
  - statsmodels
  - scikit-learn
  - h5py
  - jupyterlab
  - pillow
  - grpcio
  - protobuf
  - typing-extensions
  - six
  - traitlets
  - tornado
  - sympy
  - mpmath
  - numexpr
  - networkx
  - cmake
  - numba
  - dask
  - xarray
  - tqdm
  - psutil
  - pytest
  - plotly
  - altair
  - black
  - flake8
  - pylint
  - jupytext
  - pytorch
  - torchvision
  - torchaudio
  - autograd
  - category_encoders
  - fancyimpute
  - ipywidgets
  - flask
  - jinja2
  - optuna
  - pyyaml
  - albumentations
  - imagehash
  - tensorboard
  - tensorflow-probability
  - absl-py
  - opt-einsum
  - rich
  - termcolor
  - tensorflow-datasets
  - gin-config
  - opencv
  - pymc
  - keras
  - pip:
      - tensorflow-macos
      - tensorflow-metal
      - tensorflow-addons
      - pgmpy
      - asitop
```

---

### docs/data_loader_old.py

```python
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
```

---

### docs/model_factory_old.py

```python
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
            raise RuntimeError(error_msg) from e```

---

### docs/test_data_loader.py

```python
"""
Test script for the new data_loader.py implementation - structure verification only.
This script tests the structure of the data_loader.py and related files without
requiring TensorFlow to be installed.
"""

import os
import sys
import inspect
import importlib.util

# Add parent directory to path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_file_for_function(file_path, function_name):
    """Check if a file contains a function with the given name."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            return f"def {function_name}" in content
    except Exception as e:
        return False

def check_file_for_class_method(file_path, class_name, method_name):
    """Check if a file contains a class with the given method."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            # This is a very basic check - it might have false positives or negatives
            class_index = content.find(f"class {class_name}")
            if class_index == -1:
                return False
            method_index = content.find(f"def {method_name}", class_index)
            return method_index != -1
    except Exception as e:
        return False

print("Checking structure of data_loader.py and related files...")

# Check that files exist
data_loader_path = "/Users/jeremy/Documents/Development/plant2/src/preprocessing/data_loader.py"
dataset_loader_path = "/Users/jeremy/Documents/Development/plant2/src/preprocessing/dataset_loader.py"
dataset_pipeline_path = "/Users/jeremy/Documents/Development/plant2/src/preprocessing/dataset_pipeline.py"
data_transformations_path = "/Users/jeremy/Documents/Development/plant2/src/preprocessing/data_transformations.py"

print("\nChecking file existence:")
for path, name in [
    (data_loader_path, "data_loader.py"),
    (dataset_loader_path, "dataset_loader.py"),
    (dataset_pipeline_path, "dataset_pipeline.py"),
    (data_transformations_path, "data_transformations.py")
]:
    if os.path.exists(path):
        print(f"✓ {name} exists")
    else:
        print(f"✗ ERROR: {name} does not exist")

# Check for required methods in dataset_loader.py
print("\nChecking dataset_loader.py for required methods:")
methods = [
    "load_dataset_from_directory",
    "split_dataset",
    "save_dataset_splits",
    "get_class_weights"
]

for method in methods:
    if check_file_for_class_method(dataset_loader_path, "DatasetLoader", method):
        print(f"✓ DatasetLoader.{method} found")
    else:
        print(f"✗ ERROR: DatasetLoader.{method} not found")

# Check for required methods in data_loader.py
print("\nChecking data_loader.py for required methods:")
data_loader_methods = [
    "load_data",
    "get_class_weights"
]

for method in data_loader_methods:
    if check_file_for_class_method(data_loader_path, "DataLoader", method):
        print(f"✓ DataLoader.{method} found")
    else:
        print(f"✗ ERROR: DataLoader.{method} not found")

# Check for required functions in data_transformations.py
print("\nChecking data_transformations.py for required functions:")
transform_funcs = [
    "get_standard_augmentation_pipeline",
    "get_enhanced_augmentation_pipeline",
    "get_batch_augmentation_pipeline",
    "get_validation_transforms",
]

for func in transform_funcs:
    if check_file_for_function(data_transformations_path, func):
        print(f"✓ {func} found")
    else:
        print(f"✗ ERROR: {func} not found")

# Check for required methods in dataset_pipeline.py
print("\nChecking dataset_pipeline.py for required methods:")
pipeline_methods = [
    "create_training_pipeline",
    "create_validation_pipeline",
    "create_test_pipeline"
]

for method in pipeline_methods:
    if check_file_for_class_method(dataset_pipeline_path, "DatasetPipeline", method):
        print(f"✓ DatasetPipeline.{method} found")
    else:
        print(f"✗ ERROR: DatasetPipeline.{method} not found")

print("\nStructure check complete!")
print("If all checks passed, the data_loader.py and related files should be properly structured.")
print("You can now use data_loader.py in your main code once TensorFlow is installed.")```

---

### setup.py

```python
from setuptools import setup, find_packages

setup(
    name="deepcropdx",
    version="1.0.0",
    description="Deep Learning Models for Plant Disease Detection",
    author="Jeremy Cleland",
    author_email="jeremy.cleland@icloud.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "tensorflow>=2.7.0",
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "tqdm>=4.60.0",
        "pillow>=8.2.0",
        "jinja2>=3.0.0",
        "pyyaml>=6.0.0",
    ],
    entry_points={
        "console_scripts": [
            "deepcropdx-batch=src.main:main",
        ],
    },
    python_requires=">=3.8",
)
```

---

### src/config/__init__.py

```python
```

---

### src/config/config.py

```python
import os
from pathlib import Path


class ProjectPaths:
    def __init__(self, base_dir=None):
        """Initialize project paths.

        Args:
            base_dir: Base directory of the project. If None, uses the parent directory of this file.
        """
        if base_dir is None:
            # Get the absolute path of the parent directory
            self.base_dir = Path(__file__).parent.parent.parent.absolute()
        else:
            self.base_dir = Path(base_dir).absolute()

        # Source code directories
        self.src_dir = self.base_dir / "src"
        self.config_dir = self.src_dir / "config"
        self.model_configs_dir = self.config_dir / "model_configs"
        self.models_dir = self.src_dir / "models"
        self.preprocessing_dir = self.src_dir / "preprocessing"
        self.evaluation_dir = self.src_dir / "evaluation"
        self.training_dir = self.src_dir / "training"
        self.utils_dir = self.src_dir / "utils"
        self.scripts_dir = self.src_dir / "scripts"

        # Data directories
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"

        # Model output directories
        self.trials_dir = self.base_dir / "trials"

        # Logs directory
        self.logs_dir = self.base_dir / "logs"

        # Ensure critical directories exist
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """Create all necessary directories if they don't exist"""
        directories = [
            self.src_dir,
            self.config_dir,
            self.model_configs_dir,
            self.models_dir,
            self.preprocessing_dir,
            self.evaluation_dir,
            self.training_dir,
            self.utils_dir,
            self.scripts_dir,
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.trials_dir,
            self.logs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_trial_dir(self, model_name, run_id=None):
        """Get the trial directory for a specific model.

        Args:
            model_name: Name of the model (e.g., "EfficientNetB1")
            run_id: Specific run ID. If None, will use a timestamp.

        Returns:
            Path to the model trial directory
        """
        from datetime import datetime

        model_dir = self.trials_dir / model_name

        if run_id is None:
            # Generate a timestamped run ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Find the latest run number
            existing_runs = [
                d for d in model_dir.glob(f"run_{timestamp}_*") if d.is_dir()
            ]
            if existing_runs:
                latest_num = max([int(d.name.split("_")[-1]) for d in existing_runs])
                run_id = f"run_{timestamp}_{(latest_num + 1):03d}"
            else:
                run_id = f"run_{timestamp}_001"

        run_dir = model_dir / run_id

        # Create subdirectories for training and evaluation
        train_dir = run_dir / "training"
        eval_dir = run_dir / "evaluation"
        checkpoints_dir = train_dir / "checkpoints"
        plots_dir = train_dir / "plots"
        tensorboard_dir = train_dir / "tensorboard"

        # Create all directories
        for directory in [
            run_dir,
            train_dir,
            eval_dir,
            checkpoints_dir,
            plots_dir,
            tensorboard_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        return run_dir

    def get_config_path(self):
        """Get the path to the main configuration file"""
        return self.config_dir / "config.yaml"

    def get_model_config_path(self, model_name=None):
        """Get the path to the model configuration file.

        Args:
            model_name: Name of the model. If None, returns the models.yaml path.

        Returns:
            Path to the model configuration file
        """
        if model_name is None:
            return self.model_configs_dir / "models.yaml"

        # Try model-specific file first
        model_file = f"{model_name.lower().split('_')[0]}.yaml"
        specific_path = self.model_configs_dir / model_file

        if specific_path.exists():
            return specific_path

        # Fall back to models.yaml
        return self.model_configs_dir / "models.yaml"


# Create a singleton instance
project_paths = ProjectPaths()


def get_paths():
    """Get the project paths singleton instance"""
    return project_paths
```

---

### src/config/config.yaml

```yaml
# src/config/config.yaml

project:
  name: Plant Disease Detection
  description: Deep learning models for detecting diseases in plants
  version: 1.0.0

seed: 42

paths:
  data: 
    raw: data/raw
    processed: data/processed
  models: model_registry
  logs: trials

training:
  batch_size: 128
  epochs: 50
  learning_rate: 0.001
  optimizer: adam
  loss: categorical_crossentropy
  metrics: [accuracy, AUC, Precision, Recall]
  early_stopping:
    enabled: true
    patience: 10
    monitor: val_loss
  validation_split: 0.2
  test_split: 0.1
  progress_bar: true
  class_weight: "balanced"  # Added this line to enable class weighting
  clip_norm: 1.0 
  clip_value: null
  # Learning rate scheduler
  lr_schedule:
    enabled: true
    # Type can be: warmup_cosine, warmup_exponential, warmup_step, one_cycle, reduce_on_plateau
    type: "warmup_cosine"
    # Number of warmup epochs
    warmup_epochs: 5
    # Minimum learning rate
    min_lr: 1.0e-6
    # For reduce_on_plateau (if used)
    factor: 0.5
    patience: 5

hardware:
  use_metal: true
  mixed_precision: true
  memory_growth: true
  num_parallel_calls: 16
  prefetch_buffer_size: 8

logging:
  level: INFO
  tensorboard: true
  separate_loggers: true

reporting:
  generate_plots: true
  save_confusion_matrix: true
  save_roc_curves: true
  save_precision_recall: true
  generate_html_report: true

# Dataset configuration
data:
  image_size: [224, 224]    # Default image dimensions for preprocessing
  save_splits: true         # Whether to save dataset splits to disk
  use_saved_splits: true   # Whether to use saved splits when available
  splits_dir: "splits"      # Directory name for storing splits
  cache_parsed_images: false  # Whether to cache parsed images in memory


data_augmentation:
  enabled: true
  rotation_range: 20
  width_shift_range: 0.2
  height_shift_range: 0.2
  shear_range: 0.2
  zoom_range: 0.2
  horizontal_flip: true
  vertical_flip: false
  fill_mode: "nearest"
  
  # Augmentation settings
  color_jitter: true        # Enable color space augmentations
  gaussian_noise: true      # Add random Gaussian noise
  noise_stddev: 0.01        # Standard deviation for Gaussian noise
  random_erasing: true      # Enable random erasing (occlusion)
  erasing_prob: 0.1         # Probability of applying random erasing
  perspective_transform: true  # Enable perspective transformations
  perspective_delta: 0.1    # Max distortion for perspective transform
  
  # Batch-level augmentations
  batch_augmentations: true  # Enable batch-level augmentations
  mixup: true               # Enable MixUp augmentation
  mixup_alpha: 0.2          # Alpha parameter for MixUp
  cutmix: true              # Enable CutMix augmentation
  cutmix_alpha: 1.0         # Alpha parameter for CutMix
  
  # Validation transforms
  validation_augmentation: true    # Enable validation-time processing
  validation_resize_factor: 1.14   # Resize factor before center crop

# Add data validation settings
data_validation:
  enabled: true
  min_samples_per_class: 5
  max_class_imbalance_ratio: 10.0
  min_image_dimensions: [32, 32]
  check_corrupt_images: true
  check_duplicates: false
  max_workers: 16
  max_images_to_check: 10000


  # Model quantization settings
  quantization:
    enabled: false
    # Quantization type: "post_training" or "during_training"
    type: "post_training"
    # Quantization format: "int8", "float16", etc.
    format: "int8"
    # Whether to optimize for inference
    optimize_for_inference: true
    # Whether to measure performance after quantization
    measure_performance: true

  # Model pruning settings
  pruning:
    enabled: false
    # Pruning type: "magnitude", "structured", etc.
    type: "magnitude"
    # Target sparsity (percentage of weights to prune)
    target_sparsity: 0.5
    # Whether to perform pruning during training
    during_training: true
    # Pruning schedule: "constant" or "polynomial"
    schedule: "polynomial"
    # Start step for pruning
    start_step: 0
    # End step for pruning
    end_step: 100
    # Pruning frequency (every N steps)
    frequency: 10```

---

### src/config/config_loader.py

```python
import os
import yaml
from pathlib import Path

from src.config.config import get_paths


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
```

---

### src/config/config_manager.py

```python
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
        return self.args.attention or self.config.get("training", {}).get("attention_type")```

---

### src/config/model_configs/__init__.py

```python
```

---

### src/config/model_configs/models.yaml

```yaml
# Ideal models for plant disease detection

MobileNet:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.2
  fine_tuning:
    enabled: true
    freeze_layers: 50
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

MobileNetV2:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.2
  fine_tuning:
    enabled: true
    freeze_layers: 80
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

MobileNetV3Large:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.2
  fine_tuning:
    enabled: true
    freeze_layers: 100
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

MobileNetV3Small:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.1
  fine_tuning:
    enabled: true
    freeze_layers: 50
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

EfficientNetB0:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.2
  fine_tuning:
    enabled: true
    freeze_layers: 70
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

EfficientNetB1:
  input_shape: [240, 240, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.3
  fine_tuning:
    enabled: true
    freeze_layers: 100
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

ResNet50:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.3
  fine_tuning:
    enabled: true
    freeze_layers: 100
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

ResNet50V2:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.3
  fine_tuning:
    enabled: true
    freeze_layers: 100
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

Xception:
  input_shape: [299, 299, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.4
  fine_tuning:
    enabled: true
    freeze_layers: 120
  preprocessing:
    rescale: 1./255
    validation_augmentation: false

DenseNet121:
  input_shape: [224, 224, 3]
  include_top: false 
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.2
  fine_tuning:
    enabled: true
    freeze_layers: 100
  preprocessing:
    rescale: 1./255
    validation_augmentation: false


EfficientNetB0_SE:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.3
  attention_type: "se"
  fine_tuning:
    enabled: true
    freeze_layers: 70
    progressive: true
    finetuning_epochs: 5
  preprocessing:
    rescale: 1./255
    validation_augmentation: true
  hyperparameters:
    learning_rate: 0.0005
    batch_size: 32
    optimizer: adam

EfficientNetB1_SE:
  input_shape: [240, 240, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.3
  attention_type: "se"
  fine_tuning:
    enabled: true
    freeze_layers: 100
    progressive: true
    finetuning_epochs: 5
  preprocessing:
    rescale: 1./255
    validation_augmentation: true
  hyperparameters:
    learning_rate: 0.0003
    batch_size: 28
    optimizer: adam

EfficientNetB2_SE:
  input_shape: [260, 260, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.3
  attention_type: "se"
  fine_tuning:
    enabled: true
    freeze_layers: 120
    progressive: true
    finetuning_epochs: 5
  preprocessing:
    rescale: 1./255
    validation_augmentation: true
  hyperparameters:
    learning_rate: 0.0002
    batch_size: 24
    optimizer: adam

ResNet50_CBAM:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.3
  attention_type: "cbam"
  fine_tuning:
    enabled: true
    freeze_layers: 100
    progressive: true
    finetuning_epochs: 5
  preprocessing:
    rescale: 1./255
    validation_augmentation: true
  hyperparameters:
    learning_rate: 0.0003
    batch_size: 32
    optimizer: adam

ResNet101_CBAM:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.3
  attention_type: "cbam"
  fine_tuning:
    enabled: true
    freeze_layers: 120
    progressive: true
    finetuning_epochs: 5
  preprocessing:
    rescale: 1./255
    validation_augmentation: true
  hyperparameters:
    learning_rate: 0.0002
    batch_size: 24
    optimizer: adam

AttentionEfficientNetB0:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.4
  fine_tuning:
    enabled: true
    freeze_layers: 60
    progressive: true
    finetuning_epochs: 5
  preprocessing:
    rescale: 1./255
    validation_augmentation: true
  hyperparameters:
    learning_rate: 0.0005
    batch_size: 32
    optimizer: adam

AttentionEfficientNetB1:
  input_shape: [240, 240, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.4
  fine_tuning:
    enabled: true
    freeze_layers: 100
    progressive: true
    finetuning_epochs: 5
  preprocessing:
    rescale: 1./255
    validation_augmentation: true
  hyperparameters:
    learning_rate: 0.0003
    batch_size: 24
    optimizer: adam

AttentionResNet50:
  input_shape: [224, 224, 3]
  include_top: false
  weights: "imagenet"
  pooling: avg
  dropout_rate: 0.4
  fine_tuning:
    enabled: true
    freeze_layers: 80
    progressive: true
    finetuning_epochs: 5
  preprocessing:
    rescale: 1./255
    validation_augmentation: true
  hyperparameters:
    learning_rate: 0.0003
    batch_size: 32
    optimizer: adam
    discriminative_lr:
      enabled: true
      base_lr: 0.0003
      factor: 0.3```

---

### src/evaluation/__init__.py

```python
```

---

### src/evaluation/metrics.py

```python
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import json
from pathlib import Path
from tqdm.auto import tqdm


def calculate_metrics(y_true, y_pred, y_pred_prob=None, class_names=None):
    """
    Calculate evaluation metrics for classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities (optional)
        class_names: List of class names (optional)

    Returns:
        Dictionary with calculated metrics
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
    else:
        y_true_indices = y_true

    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_indices = np.argmax(y_pred, axis=1)
    else:
        y_pred_indices = y_pred

    # Calculate basic metrics
    metrics = {
        "accuracy": accuracy_score(y_true_indices, y_pred_indices),
        "precision_macro": precision_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            y_true_indices, y_pred_indices, average="macro", zero_division=0
        ),
        "precision_weighted": precision_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_true_indices, y_pred_indices, average="weighted", zero_division=0
        ),
    }

    # Calculate AUC if probabilities are provided
    if y_pred_prob is not None:
        # For multi-class classification
        if y_pred_prob.shape[1] > 2:
            try:
                # Convert y_true to one-hot encoding if it's not already
                if y_true.ndim == 1 or y_true.shape[1] == 1:
                    n_classes = y_pred_prob.shape[1]
                    y_true_onehot = np.zeros((len(y_true_indices), n_classes))
                    y_true_onehot[np.arange(len(y_true_indices)), y_true_indices] = 1
                else:
                    y_true_onehot = y_true

                metrics["roc_auc_macro"] = roc_auc_score(
                    y_true_onehot, y_pred_prob, average="macro", multi_class="ovr"
                )
                metrics["roc_auc_weighted"] = roc_auc_score(
                    y_true_onehot, y_pred_prob, average="weighted", multi_class="ovr"
                )
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")
        # For binary classification
        elif y_pred_prob.shape[1] == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true_indices, y_pred_prob[:, 1])
            except Exception as e:
                print(f"Warning: Could not calculate ROC AUC: {e}")

    # Add per-class metrics if class names are provided
    if class_names is not None:
        # Get report as dictionary
        report = classification_report(
            y_true_indices,
            y_pred_indices,
            output_dict=True,
            target_names=(
                class_names
                if isinstance(class_names, list)
                else list(class_names.values())
            ),
        )

        # Add per-class metrics to the main metrics dictionary
        metrics["per_class"] = {}
        for class_name in report:
            if class_name not in ["accuracy", "macro avg", "weighted avg"]:
                metrics["per_class"][class_name] = {
                    "precision": report[class_name]["precision"],
                    "recall": report[class_name]["recall"],
                    "f1-score": report[class_name]["f1-score"],
                    "support": report[class_name]["support"],
                }

    # Calculate confusion matrix but don't include in returned metrics
    # (it's not JSON serializable)
    cm = confusion_matrix(y_true_indices, y_pred_indices)

    return metrics


def evaluate_model(
    model, test_data, class_names=None, metrics_path=None, use_tqdm=True
):
    """
    Evaluate a model on test data with progress tracking

    Args:
        model: TensorFlow model to evaluate
        test_data: Test dataset
        class_names: List of class names (optional)
        metrics_path: Path to save metrics (optional)
        use_tqdm: Whether to use tqdm progress bar

    Returns:
        Dictionary with evaluation metrics
    """
    # Create predictions with progress bar
    print("Generating predictions...")

    if use_tqdm:
        # Get number of batches
        n_batches = len(test_data)

        # Initialize lists for predictions and true labels
        all_y_pred = []
        all_y_true = []

        # Use tqdm for progress tracking
        for batch_idx, (x, y) in enumerate(
            tqdm(test_data, desc="Predicting", total=n_batches)
        ):
            # Get predictions for this batch
            y_pred = model.predict(x, verbose=0)
            all_y_pred.append(y_pred)
            all_y_true.append(y)

        # Concatenate all batches
        y_pred_prob = np.vstack(all_y_pred)
        y_true = np.vstack(all_y_true)
    else:
        # Standard evaluation without progress tracking
        y_pred_prob = model.predict(test_data)
        y_true = np.concatenate([y for x, y in test_data], axis=0)

    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_prob, class_names)

    # Add standard evaluation metrics
    if hasattr(model, "evaluate"):
        results = model.evaluate(test_data, verbose=1)
        for i, metric_name in enumerate(model.metrics_names):
            metrics[metric_name] = results[i]

    # Save metrics if path is provided
    if metrics_path:
        # Convert path to Path object if it's a string
        if isinstance(metrics_path, str):
            metrics_path = Path(metrics_path)

        # Create parent directories if they don't exist
        metrics_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python native types for JSON serialization
        metrics_json = {}
        for key, value in metrics.items():
            if key == "per_class":
                metrics_json[key] = {}
                for class_name, class_metrics in value.items():
                    metrics_json[key][class_name] = {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                        for k, v in class_metrics.items()
                    }
            else:
                metrics_json[key] = (
                    float(value)
                    if isinstance(value, (np.float32, np.float64))
                    else value
                )

        # Save to file
        with open(metrics_path, "w") as f:
            json.dump(metrics_json, f, indent=4)

    return metrics
```

---

### src/evaluation/visualization.py

```python
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from pathlib import Path


def plot_training_history(history, save_path=None):
    """Plot training and validation metrics over epochs"""
    plt.figure(figsize=(12, 5))

    # Plot accuracy
    plt.subplot(1, 2, 1)
    if "accuracy" in history.history:
        plt.plot(history.history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in history.history:
            plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
    elif "acc" in history.history:  # For compatibility with older TF versions
        plt.plot(history.history["acc"], label="Training Accuracy")
        if "val_acc" in history.history:
            plt.plot(history.history["val_acc"], label="Validation Accuracy")
        plt.title("Model Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    if "val_loss" in history.history:
        plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    y_true, y_pred, class_names=None, save_path=None, figsize=(10, 8), normalize=False
):
    """
    Plot confusion matrix

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
        normalize: Whether to normalize the confusion matrix
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Normalize confusion matrix if requested
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(cm.shape[0])]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(cm.shape[0])]
    else:
        labels = class_names

    # Truncate long class names
    max_length = 20
    labels = [
        label[:max_length] + "..." if len(label) > max_length else label
        for label in labels
    ]

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    # Add counts to plot title
    plt.figtext(0.5, 0.01, f"Total samples: {len(y_true)}", ha="center")

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_roc_curve(
    y_true, y_pred_prob, class_names=None, save_path=None, figsize=(10, 8)
):
    """
    Plot ROC curve for multi-class classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
        y_true_onehot = y_true
    else:
        y_true_indices = y_true
        # Convert to one-hot encoding
        n_classes = y_pred_prob.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Calculate ROC curve and ROC area for each class
    n_classes = y_pred_prob.shape[1]

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(n_classes)]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(n_classes)]
    else:
        labels = class_names

    plt.figure(figsize=figsize)

    # Plot ROC curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{labels[i]} (AUC = {roc_auc:.2f})")

    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_precision_recall_curve(
    y_true, y_pred_prob, class_names=None, save_path=None, figsize=(10, 8)
):
    """
    Plot precision-recall curve for multi-class classification

    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred_prob: Predicted probabilities
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true_indices = np.argmax(y_true, axis=1)
        y_true_onehot = y_true
    else:
        y_true_indices = y_true
        # Convert to one-hot encoding
        n_classes = y_pred_prob.shape[1]
        y_true_onehot = np.zeros((len(y_true), n_classes))
        y_true_onehot[np.arange(len(y_true)), y_true] = 1

    # Calculate Precision-Recall curve for each class
    n_classes = y_pred_prob.shape[1]

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in range(n_classes)]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in range(n_classes)]
    else:
        labels = class_names

    plt.figure(figsize=figsize)

    # Plot Precision-Recall curve for each class
    for i in range(n_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_onehot[:, i], y_pred_prob[:, i]
        )
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, lw=2, label=f"{labels[i]} (AUC = {pr_auc:.2f})")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_class_distribution(y_true, class_names=None, save_path=None, figsize=(12, 6)):
    """
    Plot class distribution

    Args:
        y_true: True labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)

    # Count occurrences of each class
    unique_classes, counts = np.unique(y_true, return_counts=True)

    # Process class names
    if class_names is None:
        # Use indices as class names
        labels = [str(i) for i in unique_classes]
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, convert it to a list of names
        labels = [class_names[i] for i in unique_classes]
    else:
        labels = [class_names[i] for i in unique_classes]

    # Sort by frequency
    idx = np.argsort(counts)[::-1]
    counts = counts[idx]
    labels = [labels[i] for i in idx]

    # Plot
    plt.figure(figsize=figsize)
    bars = plt.bar(range(len(counts)), counts, align="center")
    plt.xticks(range(len(counts)), labels, rotation=45, ha="right")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Class Distribution")

    # Add counts on top of bars
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 5,
            str(counts[i]),
            ha="center",
        )

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_misclassified_examples(
    x_test,
    y_true,
    y_pred,
    class_names=None,
    num_examples=9,
    save_path=None,
    figsize=(15, 15),
):
    """
    Plot misclassified examples

    Args:
        x_test: Test images
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted labels (one-hot encoded or class indices)
        class_names: List or dictionary of class names
        num_examples: Number of examples to plot
        save_path: Path to save the plot
        figsize: Figure size
    """
    # Convert one-hot encoded labels to class indices if needed
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Find misclassified examples
    misclassified = np.where(y_true != y_pred)[0]

    if len(misclassified) == 0:
        print("No misclassified examples found!")
        return

    # Process class names
    if class_names is None:
        # Use indices as class names
        get_class_name = lambda idx: str(idx)
    elif isinstance(class_names, dict):
        # If class_names is a dictionary, use it directly
        get_class_name = lambda idx: class_names[idx]
    else:
        # If class_names is a list, use indices
        get_class_name = lambda idx: class_names[idx]

    # Select random misclassified examples
    indices = np.random.choice(
        misclassified, size=min(num_examples, len(misclassified)), replace=False
    )

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(len(indices))))

    # Plot
    fig, axes = plt.subplots(grid_size, grid_size, figsize=figsize)
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        # Get the image
        img = x_test[idx]

        # For greyscale images
        if img.shape[-1] == 1:
            img = img.reshape(img.shape[:-1])

        # Normalize image if needed
        if img.max() > 1.0:
            img = img / 255.0

        # Plot the image
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {get_class_name(y_true[idx])}\nPred: {get_class_name(y_pred[idx])}"
        )
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        # Convert to Path if it's a string
        if isinstance(save_path, str):
            save_path = Path(save_path)

        # Create parent directories if they don't exist
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
```

---

### src/main.py

```python
#!/usr/bin/env python3
"""
Main module for the plant disease detection model training system.
This module provides a command-line interface for training plant disease detection models.
"""

import tensorflow as tf
from typing import Dict, List, Any, Optional

from src.utils.cli_utils import handle_cli_args, get_project_info
from src.utils.hardware_utils import configure_hardware, print_hardware_summary
from src.training.training_pipeline import (
    execute_training_pipeline,
    generate_training_reports,
    clean_up_resources
)
from src.utils.error_handling import handle_exception


def main() -> int:
    """
    Main entry point for the plant disease detection training system.
    
    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Handle command line arguments and load configuration
        config_manager, config, should_print_hardware = handle_cli_args()

        # Print hardware summary and exit if requested
        if should_print_hardware:
            print_hardware_summary()
            return 0

        # Configure hardware
        hardware_info = configure_hardware(config)

        # Get project info and print startup message
        project_name, project_version = get_project_info(config)
        print(f"Starting {project_name} v{project_version} Batch Training")

        # Execute the training pipeline
        batch_trainer, total_time, exit_code = execute_training_pipeline(
            config, 
            config_manager, 
            hardware_info
        )
        
        # Generate reports if training was successful
        if batch_trainer and exit_code == 0:
            generate_training_reports(batch_trainer, total_time)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 1
    except Exception as e:
        handle_exception(e, "Error in main process")
        return 1
    finally:
        # Always clean up resources at the end
        clean_up_resources()
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
```

---

### src/model_registry/__init__.py

```python
"""
Model Registry for tracking and managing trained models.

This module provides functionality for:
- Registering and tracking trained models
- Managing model versions and runs
- Comparing model performance
- Generating reports and visualizations
"""

from src.model_registry.registry_manager import ModelRegistryManager

__all__ = ["ModelRegistryManager"]
```

---

### src/model_registry/registry_manager.py

```python
import os
import json
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from src.config.config import get_paths


class ModelRegistryManager:
    """
    Manager for the model registry that keeps track of all trained models
    in the trials folder, providing methods to register, retrieve, and compare models.
    """

    def __init__(self):
        """Initialize the model registry manager"""
        self.paths = get_paths()
        self.registry_file = self.paths.trials_dir / "registry.json"
        self._registry = self._load_registry()

    def _load_registry(self):
        """Load the registry from the JSON file or create a new one"""
        if self.registry_file.exists():
            with open(self.registry_file, "r") as f:
                return json.load(f)
        else:
            # Create default registry structure
            registry = {
                "models": {},
                "metadata": {
                    "version": "1.0.0",
                    "last_updated": datetime.now().isoformat(),
                    "total_models": 0,
                    "total_runs": 0,
                },
            }
            self._save_registry(registry)
            return registry

    def _save_registry(self, registry=None):
        """Save the registry to the JSON file"""
        if registry is None:
            registry = self._registry

        # Update metadata
        registry["metadata"]["last_updated"] = datetime.now().isoformat()
        registry["metadata"]["total_models"] = len(registry["models"])
        registry["metadata"]["total_runs"] = sum(
            len(model_info["runs"]) for model_info in registry["models"].values()
        )

        # Ensure the directory exists
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(self.registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def scan_trials(self, rescan=False):
        """
        Scan the trials directory to discover models and runs that aren't
        in the registry yet.

        Args:
            rescan: If True, rescan all model directories even if they're in the registry

        Returns:
            Number of new runs added to the registry
        """
        trials_dir = self.paths.trials_dir
        if not trials_dir.exists():
            print(f"Trials directory {trials_dir} doesn't exist. Creating...")
            trials_dir.mkdir(parents=True, exist_ok=True)
            return 0

        # Get all model directories
        model_dirs = [
            d for d in trials_dir.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]

        # Initialize counters
        new_models = 0
        new_runs = 0

        # Iterate through model directories
        for model_dir in tqdm(model_dirs, desc="Scanning models"):
            model_name = model_dir.name

            # Skip if already in registry and not rescanning
            if not rescan and model_name in self._registry["models"]:
                continue

            # Add model to registry if needed
            if model_name not in self._registry["models"]:
                self._registry["models"][model_name] = {
                    "name": model_name,
                    "runs": {},
                    "best_run": None,
                    "last_run": None,
                    "total_runs": 0,
                }
                new_models += 1

            # Get all run directories
            run_dirs = [
                d
                for d in model_dir.iterdir()
                if d.is_dir() and d.name.startswith("run_")
            ]

            # Iterate through run directories
            for run_dir in run_dirs:
                run_id = run_dir.name

                # Skip if already in registry and not rescanning
                if (
                    not rescan
                    and run_id in self._registry["models"][model_name]["runs"]
                ):
                    continue

                # Extract run information
                run_info = self._extract_run_info(model_name, run_id, run_dir)

                # Add to registry
                self._registry["models"][model_name]["runs"][run_id] = run_info
                self._registry["models"][model_name]["total_runs"] += 1
                new_runs += 1

                # Update last run
                self._registry["models"][model_name]["last_run"] = run_id

                # Update best run if needed
                if self._registry["models"][model_name]["best_run"] is None:
                    self._registry["models"][model_name]["best_run"] = run_id
                else:
                    best_run_id = self._registry["models"][model_name]["best_run"]
                    best_run = self._registry["models"][model_name]["runs"][best_run_id]
                    current_accuracy = best_run.get("metrics", {}).get(
                        "test_accuracy", 0
                    )
                    new_accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)

                    if new_accuracy > current_accuracy:
                        self._registry["models"][model_name]["best_run"] = run_id

        # Save registry if any changes were made
        if new_models > 0 or new_runs > 0:
            self._save_registry()
            print(f"Added {new_models} new models and {new_runs} new runs to registry")
        else:
            print("No new models or runs found")

        return new_runs

    def _extract_run_info(self, model_name, run_id, run_dir):
        """Extract information about a model run"""
        run_info = {
            "id": run_id,
            "path": str(run_dir),
            "timestamp": None,
            "metrics": {},
            "model_path": None,
            "has_checkpoints": False,
            "has_tensorboard": False,
            "status": "unknown",
        }

        # Extract timestamp from run_id
        try:
            timestamp_part = run_id.split("_")[1:3]
            run_info["timestamp"] = "_".join(timestamp_part)
        except:
            pass

        # Check for model file
        model_file = run_dir / f"{model_name}_final.h5"
        if model_file.exists():
            run_info["model_path"] = str(model_file)
            run_info["status"] = "completed"

        # Check for metrics file
        metrics_file = run_dir / "final_metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file, "r") as f:
                    run_info["metrics"] = json.load(f)
            except:
                pass

        # Check for evaluation metrics
        eval_metrics_file = run_dir / "evaluation" / "metrics.json"
        if eval_metrics_file.exists():
            try:
                with open(eval_metrics_file, "r") as f:
                    eval_metrics = json.load(f)
                    # Add evaluation metrics with "eval_" prefix to avoid conflicts
                    for key, value in eval_metrics.items():
                        if key not in run_info["metrics"]:
                            # Only add if not already present
                            run_info["metrics"][f"eval_{key}"] = value
            except:
                pass

        # Check for checkpoints
        checkpoint_dir = run_dir / "training" / "checkpoints"
        if checkpoint_dir.exists() and any(checkpoint_dir.iterdir()):
            run_info["has_checkpoints"] = True

        # Check for tensorboard logs
        tensorboard_dir = run_dir / "training" / "tensorboard"
        if tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
            run_info["has_tensorboard"] = True

        return run_info

    def register_model(self, model, model_name, metrics, history, run_dir):
        """
        Register a trained model in the registry

        Args:
            model: The trained TensorFlow model
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            history: Training history object
            run_dir: Directory where the model is saved

        Returns:
            Run ID of the registered model
        """
        # Convert run_dir to Path if it's a string
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        # Get run_id from directory name
        run_id = run_dir.name

        # Add model to registry if needed
        if model_name not in self._registry["models"]:
            self._registry["models"][model_name] = {
                "name": model_name,
                "runs": {},
                "best_run": None,
                "last_run": None,
                "total_runs": 0,
            }

        # Save the model if it hasn't been saved already
        model_path = run_dir / f"{model_name}_final.h5"
        if not model_path.exists():
            model.save(model_path)
            print(f"Model saved to {model_path}")

        # Save metrics to file if not already saved
        metrics_path = run_dir / "final_metrics.json"
        if not metrics_path.exists():
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)

        # Extract run information
        run_info = self._extract_run_info(model_name, run_id, run_dir)

        # Add to registry
        self._registry["models"][model_name]["runs"][run_id] = run_info
        self._registry["models"][model_name]["total_runs"] += 1
        self._registry["models"][model_name]["last_run"] = run_id

        # Update best run if needed
        if self._registry["models"][model_name]["best_run"] is None:
            self._registry["models"][model_name]["best_run"] = run_id
        else:
            best_run_id = self._registry["models"][model_name]["best_run"]
            best_run = self._registry["models"][model_name]["runs"][best_run_id]
            current_accuracy = best_run.get("metrics", {}).get("test_accuracy", 0)
            new_accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)

            if new_accuracy > current_accuracy:
                self._registry["models"][model_name]["best_run"] = run_id

        # Save registry
        self._save_registry()

        return run_id

    def get_model(self, model_name, run_id=None, best=False):
        """
        Get a model from the registry

        Args:
            model_name: Name of the model
            run_id: ID of the run to retrieve. If None, uses the latest run.
            best: If True, retrieves the best run instead of the latest

        Returns:
            Loaded TensorFlow model, or None if not found
        """
        # Check if model exists in registry
        if model_name not in self._registry["models"]:
            print(f"Model {model_name} not found in registry")
            return None

        # Determine run_id to retrieve
        if run_id is None:
            if best:
                run_id = self._registry["models"][model_name]["best_run"]
                if run_id is None:
                    print(f"No best run found for model {model_name}")
                    return None
            else:
                run_id = self._registry["models"][model_name]["last_run"]
                if run_id is None:
                    print(f"No runs found for model {model_name}")
                    return None
        elif run_id not in self._registry["models"][model_name]["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return None

        # Get model path
        run_info = self._registry["models"][model_name]["runs"][run_id]
        model_path = run_info.get("model_path")

        if model_path is None or not os.path.exists(model_path):
            print(f"Model file not found for {model_name} run {run_id}")
            return None

        # Load and return the model
        try:
            print(f"Loading model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully")
            return model
        except Exception as e:
            print(f"Error loading model {model_name} run {run_id}: {e}")
            return None

    def get_run_info(self, model_name, run_id=None, best=False):
        """
        Get information about a specific run

        Args:
            model_name: Name of the model
            run_id: ID of the run to retrieve. If None, uses the latest run.
            best: If True, retrieves the best run instead of the latest

        Returns:
            Dictionary with run information, or None if not found
        """
        # Check if model exists in registry
        if model_name not in self._registry["models"]:
            print(f"Model {model_name} not found in registry")
            return None

        # Determine run_id to retrieve
        if run_id is None:
            if best:
                run_id = self._registry["models"][model_name]["best_run"]
                if run_id is None:
                    print(f"No best run found for model {model_name}")
                    return None
            else:
                run_id = self._registry["models"][model_name]["last_run"]
                if run_id is None:
                    print(f"No runs found for model {model_name}")
                    return None
        elif run_id not in self._registry["models"][model_name]["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return None

        # Return run information
        return self._registry["models"][model_name]["runs"][run_id]

    def list_models(self):
        """
        List all models in the registry

        Returns:
            List of model names
        """
        return list(self._registry["models"].keys())

    def list_runs(self, model_name):
        """
        List all runs for a specific model

        Args:
            model_name: Name of the model

        Returns:
            List of run IDs, or empty list if model not found
        """
        if model_name not in self._registry["models"]:
            print(f"Model {model_name} not found in registry")
            return []

        return list(self._registry["models"][model_name]["runs"].keys())

    def get_best_models(self, top_n=5, metric="test_accuracy"):
        """
        Get the best performing models according to a specific metric

        Args:
            top_n: Number of top models to return
            metric: Metric to use for ranking

        Returns:
            List of dictionaries with model information
        """
        # Collect best run for each model
        best_models = []

        for model_name, model_info in self._registry["models"].items():
            best_run_id = model_info["best_run"]
            if best_run_id is None:
                continue

            best_run = model_info["runs"][best_run_id]
            metric_value = best_run.get("metrics", {}).get(metric, 0)

            best_models.append(
                {
                    "name": model_name,
                    "run_id": best_run_id,
                    "metric": metric,
                    "value": metric_value,
                    "path": best_run.get("path"),
                    "timestamp": best_run.get("timestamp"),
                }
            )

        # Sort by metric value (highest first)
        best_models.sort(key=lambda x: x["value"], reverse=True)

        # Return top N
        return best_models[:top_n]

    def compare_models(
        self, model_names=None, metrics=None, plot=True, output_dir=None
    ):
        """
        Compare multiple models based on specified metrics

        Args:
            model_names: List of model names to compare. If None, uses all models.
            metrics: List of metrics to compare. If None, uses a default set.
            plot: Whether to generate comparison plots
            output_dir: Directory to save plots. If None, uses trials/comparisons.

        Returns:
            DataFrame with comparison results
        """
        # Set default metrics if none provided
        if metrics is None:
            metrics = ["test_accuracy", "test_loss", "training_time"]

        # Use all models if none specified
        if model_names is None:
            model_names = self.list_models()

        # Prepare data for comparison
        comparison_data = []

        for model_name in model_names:
            if model_name not in self._registry["models"]:
                print(f"Model {model_name} not found in registry")
                continue

            # Get best run
            best_run_id = self._registry["models"][model_name]["best_run"]
            if best_run_id is None:
                print(f"No best run found for model {model_name}")
                continue

            best_run = self._registry["models"][model_name]["runs"][best_run_id]

            # Collect metrics
            model_data = {
                "Model": model_name,
                "Run ID": best_run_id,
            }

            # Add each requested metric
            for metric in metrics:
                model_data[metric] = best_run.get("metrics", {}).get(metric, None)

            comparison_data.append(model_data)

        # Create DataFrame
        comparison_df = pd.DataFrame(comparison_data)

        # Generate plots if requested
        if plot and len(comparison_data) > 0:
            if output_dir is None:
                output_dir = self.paths.trials_dir / "comparisons"

            # Ensure directory exists
            Path(output_dir).mkdir(parents=True, exist_ok=True)

            # Plot each metric
            for metric in metrics:
                if metric in comparison_df.columns:
                    plt.figure(figsize=(12, 6))

                    # Sort by metric value
                    sorted_df = comparison_df.sort_values(by=metric, ascending=False)

                    # Create bar plot
                    ax = sns.barplot(x="Model", y=metric, data=sorted_df)

                    # Add values on top of bars
                    for i, v in enumerate(sorted_df[metric]):
                        if v is not None:
                            ax.text(i, v, f"{v:.4f}", ha="center")

                    plt.title(f"Model Comparison - {metric}")
                    plt.xticks(rotation=45, ha="right")
                    plt.tight_layout()

                    # Save plot
                    plt.savefig(Path(output_dir) / f"comparison_{metric}.png")
                    plt.close()

            # Create a combined metrics plot
            plt.figure(figsize=(14, 8))

            # Number of metrics to plot
            valid_metrics = [m for m in metrics if m in comparison_df.columns]
            n_metrics = len(valid_metrics)

            if n_metrics > 0:
                # Normalize metrics for combined visualization
                normalized_df = comparison_df.copy()

                for metric in valid_metrics:
                    if metric in normalized_df.columns:
                        values = normalized_df[metric].dropna()
                        if len(values) > 0:
                            min_val = values.min()
                            max_val = values.max()
                            if max_val > min_val:
                                normalized_df[f"{metric}_norm"] = (
                                    normalized_df[metric] - min_val
                                ) / (max_val - min_val)
                            else:
                                normalized_df[f"{metric}_norm"] = 0

                # Plot normalized metrics
                plt.subplot(2, 1, 1)

                for i, metric in enumerate(valid_metrics):
                    norm_metric = f"{metric}_norm"
                    if norm_metric in normalized_df.columns:
                        plt.plot(
                            normalized_df["Model"],
                            normalized_df[norm_metric],
                            marker="o",
                            label=metric,
                        )

                plt.title("Normalized Metrics Comparison")
                plt.xticks(rotation=45, ha="right")
                plt.ylim(0, 1.1)
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Plot actual metrics in subplots
                for i, metric in enumerate(valid_metrics):
                    plt.subplot(2, n_metrics, n_metrics + i + 1)
                    if metric in comparison_df.columns:
                        sorted_df = comparison_df.sort_values(
                            by=metric, ascending=False
                        )
                        ax = sns.barplot(x="Model", y=metric, data=sorted_df)
                        plt.title(metric)
                        plt.xticks(rotation=45, ha="right")
                        plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig(Path(output_dir) / "comparison_combined.png")
                plt.close()

        return comparison_df

    def delete_run(self, model_name, run_id, delete_files=False):
        """
        Delete a run from the registry

        Args:
            model_name: Name of the model
            run_id: ID of the run to delete
            delete_files: Whether to delete the run's files from disk

        Returns:
            True if successful, False otherwise
        """
        # Check if model exists in registry
        if model_name not in self._registry["models"]:
            print(f"Model {model_name} not found in registry")
            return False

        # Check if run exists
        if run_id not in self._registry["models"][model_name]["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return False

        # Get run information
        run_info = self._registry["models"][model_name]["runs"][run_id]
        run_path = run_info.get("path")

        # Delete files if requested
        if delete_files and run_path and os.path.exists(run_path):
            try:
                shutil.rmtree(run_path)
                print(f"Deleted run files at {run_path}")
            except Exception as e:
                print(f"Error deleting run files: {e}")
                return False

        # Remove run from registry
        del self._registry["models"][model_name]["runs"][run_id]
        self._registry["models"][model_name]["total_runs"] -= 1

        # Update best and last run
        if self._registry["models"][model_name]["best_run"] == run_id:
            # Find new best run
            best_run_id = None
            best_accuracy = -1

            for rid, rinfo in self._registry["models"][model_name]["runs"].items():
                accuracy = rinfo.get("metrics", {}).get("test_accuracy", 0)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_run_id = rid

            self._registry["models"][model_name]["best_run"] = best_run_id

        if self._registry["models"][model_name]["last_run"] == run_id:
            # Find new last run (most recent timestamp)
            last_run_id = None
            last_timestamp = ""

            for rid, rinfo in self._registry["models"][model_name]["runs"].items():
                timestamp = rinfo.get("timestamp", "")
                if timestamp > last_timestamp:
                    last_timestamp = timestamp
                    last_run_id = rid

            self._registry["models"][model_name]["last_run"] = last_run_id

        # Delete model if no runs left
        if len(self._registry["models"][model_name]["runs"]) == 0:
            del self._registry["models"][model_name]

        # Save registry
        self._save_registry()

        return True

    def export_registry(self, output_path=None):
        """
        Export the registry to a JSON file

        Args:
            output_path: Path to save the exported registry. If None, uses a timestamped name.

        Returns:
            Path to the exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.paths.trials_dir / f"registry_export_{timestamp}.json"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save registry
        with open(output_path, "w") as f:
            json.dump(self._registry, f, indent=2)

        print(f"Registry exported to {output_path}")
        return output_path

    def import_registry(self, input_path, merge=True):
        """
        Import a registry from a JSON file

        Args:
            input_path: Path to the registry file
            merge: Whether to merge with existing registry or replace it

        Returns:
            True if successful, False otherwise
        """
        if not os.path.exists(input_path):
            print(f"Registry file {input_path} not found")
            return False

        try:
            with open(input_path, "r") as f:
                imported_registry = json.load(f)

            if merge:
                # Merge with existing registry
                for model_name, model_info in imported_registry["models"].items():
                    if model_name not in self._registry["models"]:
                        self._registry["models"][model_name] = model_info
                    else:
                        # Merge runs
                        for run_id, run_info in model_info["runs"].items():
                            if (
                                run_id
                                not in self._registry["models"][model_name]["runs"]
                            ):
                                self._registry["models"][model_name]["runs"][
                                    run_id
                                ] = run_info
                                self._registry["models"][model_name]["total_runs"] += 1
            else:
                # Replace existing registry
                self._registry = imported_registry

            # Save registry
            self._save_registry()

            print(f"Registry imported successfully from {input_path}")
            return True
        except Exception as e:
            print(f"Error importing registry: {e}")
            return False

    def generate_registry_report(self, output_path=None):
        """
        Generate an HTML report summarizing the registry contents

        Args:
            output_path: Path to save the report. If None, uses a default path.

        Returns:
            Path to the generated report
        """
        if output_path is None:
            output_path = self.paths.trials_dir / "registry_report.html"

        # Ensure directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Registry Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                }}
                .card {{
                    background: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                th, td {{
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }}
                th {{
                    background-color: #f8f9fa;
                }}
                .model-card {{
                    margin-bottom: 30px;
                }}
                .best-run {{
                    background-color: #e8f4f8;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Model Registry Report</h1>
                <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                
                <div class="card">
                    <h2>Registry Summary</h2>
                    <table>
                        <tr>
                            <th>Total Models</th>
                            <td>{self._registry["metadata"]["total_models"]}</td>
                        </tr>
                        <tr>
                            <th>Total Runs</th>
                            <td>{self._registry["metadata"]["total_runs"]}</td>
                        </tr>
                        <tr>
                            <th>Last Updated</th>
                            <td>{self._registry["metadata"]["last_updated"]}</td>
                        </tr>
                    </table>
                </div>
        """

        # Get best models
        best_models = self.get_best_models(top_n=5)

        if best_models:
            html_content += """
                <div class="card">
                    <h2>Top Performing Models</h2>
                    <table>
                        <tr>
                            <th>Rank</th>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Run ID</th>
                        </tr>
            """

            for i, model in enumerate(best_models):
                html_content += f"""
                        <tr>
                            <td>{i+1}</td>
                            <td>{model["name"]}</td>
                            <td>{model["value"]:.4f}</td>
                            <td>{model["run_id"]}</td>
                        </tr>
                """

            html_content += """
                    </table>
                </div>
            """

        # Add model details
        html_content += "<h2>Model Details</h2>"

        for model_name, model_info in self._registry["models"].items():
            best_run_id = model_info["best_run"]

            html_content += f"""
                <div class="card model-card">
                    <h3>{model_name}</h3>
                    <p><strong>Total Runs:</strong> {model_info["total_runs"]}</p>
                    
                    <h4>Runs:</h4>
                    <table>
                        <tr>
                            <th>Run ID</th>
                            <th>Timestamp</th>
                            <th>Accuracy</th>
                            <th>Loss</th>
                            <th>Status</th>
                        </tr>
                    """

            # Sort runs by timestamp (recent first)
            sorted_runs = sorted(
                model_info["runs"].items(),
                key=lambda x: x[1].get("timestamp", ""),
                reverse=True,
            )

            for run_id, run_info in sorted_runs:
                # Determine if this is the best run
                is_best = run_id == best_run_id
                row_class = "best-run" if is_best else ""

                # Get metrics
                accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)
                loss = run_info.get("metrics", {}).get("test_loss", 0)
                status = run_info.get("status", "unknown")

                html_content += f"""
                        <tr class="{row_class}">
                            <td>{run_id} {' (Best)' if is_best else ''}</td>
                            <td>{run_info.get("timestamp", "")}</td>
                            <td>{accuracy:.4f}</td>
                            <td>{loss:.4f}</td>
                            <td>{status}</td>
                        </tr>
                    """

            html_content += """
                    </table>
                </div>
            """

        html_content += """
            </div>
        </body>
        </html>
        """

        # Write the report
        with open(output_path, "w") as f:
            f.write(html_content)

        print(f"Registry report generated at {output_path}")
        return output_path
```

---

### src/models/__init__.py

```python
```

---

### src/models/advanced_architectures.py

```python
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
        return None```

---

### src/models/attention.py

```python
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
        super(SpatialAttention, self).build(input_shape)

    def call(self, inputs):
        # Average pooling along channel axis
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        # Max pooling along channel axis
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)

        # Concatenate both features
        concat = tf.concat([avg_pool, max_pool], axis=-1)

        # Apply convolution
        spatial = self.conv(concat)

        # Apply attention
        output = inputs * spatial

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

    def build(self, input_shape):
        channels = input_shape[-1]
        self.gap = tf.keras.layers.GlobalAveragePooling2D()
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
        # Global average pooling
        x = self.gap(inputs)

        # MLP with bottleneck
        x = self.dense1(x)
        x = self.dense2(x)

        # Reshape to match the input tensor's shape
        x = tf.reshape(x, [-1, 1, 1, tf.shape(inputs)[-1]])

        # Apply attention
        output = inputs * x

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
        self.channel_attention = ChannelAttention(ratio=self.ratio)
        self.spatial_attention = SpatialAttention(kernel_size=self.kernel_size)
        super(CBAMBlock, self).build(input_shape)

    def call(self, inputs):
        # Channel attention
        x = self.channel_attention(inputs)

        # Spatial attention
        x = self.spatial_attention(x)

        return x

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

    # Apply attention to the output of the model
    x = attention_layer(model.output)

    # Create a new model
    from tensorflow.keras import Model

    enhanced_model = Model(inputs=model.input, outputs=x)

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
    channels = K.int_shape(input_tensor)[-1]

    # Squeeze operation (global average pooling)
    # Check if the input is already pooled (2D) or still has spatial dimensions (4D)
    input_shape = K.int_shape(input_tensor)
    if len(input_shape) == 2:
        # Already pooled, use as is
        x = input_tensor
    else:
        # Still has spatial dimensions, apply pooling
        x = GlobalAveragePooling2D()(input_tensor)

    # Excitation operation (two FC layers with bottleneck)
    x = Dense(channels // ratio, activation="relu")(x)
    x = Dense(channels, activation="sigmoid")(x)

    # Scale the input tensor
    x = Reshape((1, 1, channels))(x)
    
    # Check if the input is already pooled (2D) or still has spatial dimensions (4D)
    input_shape = K.int_shape(input_tensor)
    if len(input_shape) == 2:
        # For already pooled input, we need to reshape both tensors to be compatible
        input_reshaped = Reshape((1, 1, channels))(input_tensor)
        x = multiply([input_reshaped, x])
        # Flatten back to match original shape
        x = Flatten()(x)
    else:
        # For spatial inputs, apply scaling as usual
        x = multiply([input_tensor, x])

    return x


class ResidualAttention(tf.keras.layers.Layer):
    """
    Residual Attention module for enhancing ResNet-like architectures.

    This implements channel attention similar to SE blocks but with a
    residual connection to maintain gradient flow.
    """

    def __init__(self, channels, reduction=16):
        """
        Initialize the Residual Attention module.

        Args:
            channels: Number of input channels
            reduction: Reduction ratio for the bottleneck
        """
        super(ResidualAttention, self).__init__()
        self.avg_pool = GlobalAveragePooling2D()
        self.dense1 = Dense(channels // reduction, activation="relu")
        self.dense2 = Dense(channels, activation="sigmoid")
        self.reshape = Reshape((1, 1, channels))

    def call(self, inputs):
        """
        Forward pass for the residual attention module.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor with attention applied
        """
        b, h, w, c = inputs.shape
        y = self.avg_pool(inputs)
        y = self.dense1(y)
        y = self.dense2(y)
        y = self.reshape(y)
        return inputs * y


class ECABlock(tf.keras.layers.Layer):
    """
    Efficient Channel Attention (ECA) block.

    This is a more lightweight alternative to SE blocks that uses
    1D convolutions instead of fully connected layers.
    """

    def __init__(self, kernel_size=3):
        """
        Initialize the ECA block.

        Args:
            kernel_size: Size of the 1D convolution kernel
        """
        super(ECABlock, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pool = GlobalAveragePooling2D()

    def build(self, input_shape):
        """
        Build the ECA block.

        Args:
            input_shape: Shape of the input tensor
        """
        self.channels = input_shape[-1]
        self.conv = tf.keras.layers.Conv1D(
            filters=1, kernel_size=self.kernel_size, padding="same", use_bias=False
        )
        super(ECABlock, self).build(input_shape)

    def call(self, inputs):
        """
        Forward pass for the ECA block.

        Args:
            inputs: Input tensor

        Returns:
            Output tensor with ECA applied
        """
        # Check if the input is already pooled (2D) or still has spatial dimensions (4D)
        input_shape = tf.keras.backend.int_shape(inputs)
        
        if len(input_shape) == 2:
            # Already pooled, use as is
            y = inputs
        else:
            # Global average pooling
            y = self.avg_pool(inputs)

        # Reshape to [batch, channels, 1]
        y = tf.reshape(y, [-1, 1, self.channels])

        # Apply 1D convolution
        y = self.conv(y)

        # Reshape and apply sigmoid activation
        y = tf.reshape(y, [-1, self.channels])
        y = tf.nn.sigmoid(y)

        if len(input_shape) == 2:
            # For already pooled input, multiply directly
            return inputs * y
        else:
            # Reshape to [batch, 1, 1, channels] for broadcasting
            y = tf.reshape(y, [-1, 1, 1, self.channels])
            # Scale the input tensor
            return inputs * y


def spatial_attention_block(input_tensor):
    """
    Spatial Attention Block for highlighting important spatial regions.

    Args:
        input_tensor: Input tensor to apply spatial attention to

    Returns:
        Output tensor with spatial attention applied
    """
    # Check if input has spatial dimensions
    input_shape = tf.keras.backend.int_shape(input_tensor)
    
    # If input is already flattened (no spatial dimensions), return as is
    if len(input_shape) == 2:
        return input_tensor
        
    # Compute channel-wise average and max pooling
    avg_pool = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(input_tensor, axis=-1, keepdims=True)

    # Concatenate the pooled features
    concat = tf.concat([avg_pool, max_pool], axis=-1)

    # Apply convolution to generate spatial attention map
    spatial_map = Conv2D(
        filters=1, kernel_size=7, padding="same", activation="sigmoid"
    )(concat)

    # Apply spatial attention
    return multiply([input_tensor, spatial_map])


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
    # Apply channel attention similar to SE block
    channels = K.int_shape(input_tensor)[-1]

    # Channel attention
    # Check if the input is already pooled (2D) or still has spatial dimensions (4D)
    input_shape = K.int_shape(input_tensor)
    if len(input_shape) == 2:
        # Already pooled, use as is
        avg_pool = input_tensor
        max_pool = input_tensor  # For already pooled data, we use the same values
    else:
        # Still has spatial dimensions, apply pooling
        avg_pool = GlobalAveragePooling2D()(input_tensor)
        max_pool = tf.reduce_max(input_tensor, axis=[1, 2])

    avg_pool = Dense(channels // ratio, activation="relu")(avg_pool)
    avg_pool = Dense(channels, activation="linear")(avg_pool)

    max_pool = Dense(channels // ratio, activation="relu")(max_pool)
    max_pool = Dense(channels, activation="linear")(max_pool)

    channel_attention = tf.nn.sigmoid(avg_pool + max_pool)
    channel_attention = Reshape((1, 1, channels))(channel_attention)
    
    # Check if the input is already pooled (2D) or still has spatial dimensions (4D)
    input_shape = K.int_shape(input_tensor)
    if len(input_shape) == 2:
        # For already pooled input, we need to reshape both tensors to be compatible
        input_reshaped = Reshape((1, 1, channels))(input_tensor)
        channel_refined = multiply([input_reshaped, channel_attention])
        # Flatten back to match original shape
        channel_refined = Flatten()(channel_refined)
    else:
        # For spatial inputs, apply scaling as usual
        channel_refined = multiply([input_tensor, channel_attention])

    # Spatial attention
    spatial_attention = spatial_attention_block(channel_refined)

    return spatial_attention


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

    # Define where to add CBAM attention (after each residual block)
    attention_indices = []
    for i, layer in enumerate(base_model.layers):
        if "add" in layer.name.lower():  # Find residual connections
            attention_indices.append(i)

    # Build the model with attention
    inputs = base_model.input
    x = inputs

    # Process through base model and add attention after each residual block
    for i, layer in enumerate(base_model.layers):
        x = layer(x)
        if i in attention_indices:
            x = cbam_block(x)

    # Classifier head
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    # Create and return the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return model
```

---

### src/models/model_factory.py

```python
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
```

---

### src/models/model_optimizer.py

```python
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
        
        return representative_dataset_gen```

---

### src/preprocessing/__init__.py

```python
```

---

### src/preprocessing/data_loader.py

```python
"""
Enhanced data loading module with better memory management and performance.
This refactored module separates concerns between dataset loading, transformation, and pipeline creation.
"""

import tensorflow as tf
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

from src.config.config import get_paths
from src.preprocessing.dataset_loader import DatasetLoader
from src.preprocessing.dataset_pipeline import DatasetPipeline
from src.utils.seed_utils import set_global_seeds


class DataLoader:
    """Enhanced data loader with improved memory management and performance."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data loader with given configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = get_paths()
        
        # Set random seed for reproducibility
        self.seed = self.config.get("seed", 42)
        set_global_seeds(self.seed)
        
        # Get training configuration
        training_config = config.get("training", {})
        self.validation_split = training_config.get("validation_split", 0.2)
        self.test_split = training_config.get("test_split", 0.1)
        
        # Initialize the dataset loader
        self.dataset_loader = DatasetLoader(config)
        
        # Initialize the dataset pipeline creator
        self.dataset_pipeline = DatasetPipeline(config)
    
    def load_data(
        self, 
        data_dir: Optional[str] = None, 
        use_saved_splits: Optional[bool] = None
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, str]]:
        """Load and prepare datasets for training, validation, and testing.
        
        Args:
            data_dir: Path to the dataset directory (optional)
            use_saved_splits: Whether to try loading from saved splits (optional)
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset, class_names)
            
        Raises:
            ValueError: If dataset cannot be loaded or validated
        """
        # Handle data directory path
        if data_dir is None:
            data_path_config = self.config.get("paths", {}).get("data", {})
            if isinstance(data_path_config, dict):
                data_dir = data_path_config.get("processed", "data/processed")
            else:
                data_dir = "data/processed"
        
        # Load raw file paths and labels
        file_paths, labels, class_names = self.dataset_loader.load_dataset_from_directory(
            data_dir=data_dir,
            validation_split=self.validation_split,
            test_split=self.test_split,
            use_saved_splits=use_saved_splits
        )
        
        # Split indices for train/val/test
        indices = self.dataset_loader.split_dataset(
            file_paths=file_paths,
            labels=labels,
            validation_split=self.validation_split,
            test_split=self.test_split
        )
        
        # Save splits if configured
        if self.config.get("data", {}).get("save_splits", False):
            try:
                print("Saving dataset splits for reproducibility")
                self.dataset_loader.save_dataset_splits(
                    file_paths=file_paths,
                    labels=labels,
                    class_names=class_names,
                    indices=indices,
                    output_dir=data_dir
                )
            except Exception as e:
                print(f"Failed to save dataset splits: {e}")
        
        # Create TensorFlow dataset pipelines
        num_classes = len(class_names)
        
        # Create training dataset
        train_dataset = self.dataset_pipeline.create_training_pipeline(
            file_paths=file_paths,
            labels=labels,
            indices=indices["train"],
            num_classes=num_classes
        )
        
        # Create validation dataset
        val_dataset = self.dataset_pipeline.create_validation_pipeline(
            file_paths=file_paths,
            labels=labels,
            indices=indices["val"],
            num_classes=num_classes
        )
        
        # Create test dataset
        test_dataset = self.dataset_pipeline.create_test_pipeline(
            file_paths=file_paths,
            labels=labels,
            indices=indices["test"],
            num_classes=num_classes
        )
        
        # Create a class mapping dictionary to ensure compatibility
        # Convert from int keys if needed
        cleaned_class_names = {}
        for k, v in class_names.items():
            if isinstance(k, str) and k.isdigit():
                cleaned_class_names[int(k)] = v
            else:
                cleaned_class_names[k] = v
                
        return train_dataset, val_dataset, test_dataset, cleaned_class_names
    
    def get_class_weights(self, labels: List[int], indices: List[int]) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets.
        
        Args:
            labels: List of all labels
            indices: List of indices to consider (e.g., training indices)
            
        Returns:
            Dictionary mapping class indices to weights
        """
        # Extract the labels for the given indices
        subset_labels = [labels[i] for i in indices]
        
        # Count classes
        class_counts = {}
        for label in subset_labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        # Calculate weights (inversely proportional to frequency)
        total = len(subset_labels)
        n_classes = len(class_counts)
        
        # Use balanced formula: total_samples / (n_classes * class_count)
        weights = {}
        for label, count in class_counts.items():
            weights[label] = total / (n_classes * count) if count > 0 else 1.0
        
        return weights```

---

### src/preprocessing/data_transformations.py

```python
"""
Data transformation and augmentation module for image preprocessing.
This separates concerns from the data_loader.py file to focus only on transformations.
"""

import tensorflow as tf
import math
from typing import Dict, Tuple, Any, Optional, Union, Callable

# Basic image transformations

def resize_image(
    image: tf.Tensor, 
    target_size: Tuple[int, int] = (224, 224)
) -> tf.Tensor:
    """Resize an image to the target size.
    
    Args:
        image: Input image tensor
        target_size: Target (height, width) for resizing
        
    Returns:
        Resized image tensor
    """
    return tf.image.resize(image, target_size)


def normalize_image(
    image: tf.Tensor, 
    method: str = "scale"
) -> tf.Tensor:
    """Normalize an image using various methods.
    
    Args:
        image: Input image tensor
        method: Normalization method ('scale', 'standardize', or 'centered')
        
    Returns:
        Normalized image tensor
    """
    if method == "scale":
        # Scale to [0, 1] range
        return tf.cast(image, tf.float32) / 255.0
    elif method == "standardize":
        # Standardize to mean=0, std=1
        return tf.image.per_image_standardization(image)
    elif method == "centered":
        # Scale to [-1, 1] range
        return tf.cast(image, tf.float32) / 127.5 - 1.0
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def center_crop(
    image: tf.Tensor, 
    target_size: Tuple[int, int] = (224, 224)
) -> tf.Tensor:
    """Perform center crop on an image.
    
    Args:
        image: Input image tensor
        target_size: Target (height, width) for cropping
        
    Returns:
        Center-cropped image tensor
    """
    return tf.image.resize_with_crop_or_pad(image, target_size[0], target_size[1])


# Advanced image augmentations

def apply_perspective_transform(
    image: tf.Tensor, 
    max_delta: float = 0.1
) -> tf.Tensor:
    """Apply a slight random distortion to an image (simplified version).
    
    Note: This is a simplified version that applies random brightness, contrast,
    and other transformations as a substitute for perspective transform, since
    the ComputeProjectiveTransform op may not be available in all TF versions.

    Args:
        image: A tensor of shape [height, width, channels]
        max_delta: Maximum distortion parameter (controls intensity)

    Returns:
        Transformed image tensor of the same shape
    """
    # Apply a combination of transformations instead of perspective transform
    # Start with random brightness
    image = tf.image.random_brightness(image, max_delta * 0.3)
    
    # Add random contrast
    contrast_factor = 1.0 + max_delta
    image = tf.image.random_contrast(image, 1.0/contrast_factor, contrast_factor)
    
    # Add random saturation
    saturation_factor = 1.0 + max_delta
    image = tf.image.random_saturation(image, 1.0/saturation_factor, saturation_factor)
    
    # Apply small random hue changes
    image = tf.image.random_hue(image, max_delta * 0.1)
    
    # Ensure the values stay in the valid range
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    return image


def random_erasing(
    image: tf.Tensor, 
    p: float = 0.5, 
    scale: Tuple[float, float] = (0.02, 0.2), 
    ratio: Tuple[float, float] = (0.3, 3.3), 
    value: float = 0
) -> tf.Tensor:
    """Randomly erase rectangles in the image (occlusion).

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

    # Create mask using scatter_nd
    rows = tf.range(i, i + h)
    cols = tf.range(j, j + w)
    indices = tf.meshgrid(rows, cols)
    indices = tf.stack(indices, axis=-1)
    indices = tf.reshape(indices, [-1, 2])

    # Create the mask
    mask_shape = tf.shape(image)
    mask = tf.ones(mask_shape, dtype=image.dtype)
    updates = tf.zeros([h * w, channels], dtype=image.dtype)
    mask = tf.tensor_scatter_nd_update(mask, indices, updates)

    # Apply mask to image
    erased_image = image * mask

    return erased_image


def add_gaussian_noise(
    image: tf.Tensor, 
    mean: float = 0.0, 
    stddev: float = 0.01
) -> tf.Tensor:
    """Add Gaussian noise to an image.

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


# Batch-level augmentations

def apply_mixup(
    images: tf.Tensor, 
    labels: tf.Tensor, 
    alpha: float = 0.2
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply MixUp augmentation to a batch of images and labels.

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


def apply_cutmix(
    images: tf.Tensor, 
    labels: tf.Tensor, 
    alpha: float = 1.0
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply CutMix augmentation to a batch of images and labels.

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


# Complete augmentation pipelines

def get_standard_augmentation_pipeline(
    config: Dict[str, Any] = None
) -> Callable:
    """Get a function that applies standard data augmentations.
    
    Args:
        config: Configuration dictionary with augmentation parameters
        
    Returns:
        Function that takes (image, label) and returns (augmented_image, label)
    """
    if config is None:
        config = {}
        
    # Extract augmentation parameters from config
    rotation_range = config.get("rotation_range", 20)
    width_shift_range = config.get("width_shift_range", 0.2)
    height_shift_range = config.get("height_shift_range", 0.2)
    zoom_range = config.get("zoom_range", 0.2)
    horizontal_flip = config.get("horizontal_flip", True)
    vertical_flip = config.get("vertical_flip", False)
    image_size = config.get("image_size", (224, 224))
    
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    
    def augment_image(image, label):
        """Apply data augmentation to an image."""
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
                image_width_float = tf.cast(image_width, tf.float32)
                w_pixels = tf.cast(image_width_float * width_shift_range, tf.int32)
                w_shift = tf.random.uniform(
                    shape=[], minval=-w_pixels, maxval=w_pixels, dtype=tf.int32
                )
                image = tf.roll(image, shift=w_shift, axis=1)

            if height_shift_range > 0:
                image_height_float = tf.cast(image_height, tf.float32)
                h_pixels = tf.cast(image_height_float * height_shift_range, tf.int32)
                h_shift = tf.random.uniform(
                    shape=[], minval=-h_pixels, maxval=h_pixels, dtype=tf.int32
                )
                image = tf.roll(image, shift=h_shift, axis=0)

        # Random flips
        if horizontal_flip and tf.random.uniform(shape=[]) > 0.5:
            image = tf.image.flip_left_right(image)

        if vertical_flip and tf.random.uniform(shape=[]) > 0.5:
            image = tf.image.flip_up_down(image)
            
        # Basic color augmentations
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        
        # Make sure pixel values are still in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label
    
    return augment_image


def get_enhanced_augmentation_pipeline(
    config: Dict[str, Any] = None
) -> Callable:
    """Get a function that applies enhanced data augmentations.
    
    Args:
        config: Configuration dictionary with augmentation parameters
        
    Returns:
        Function that takes (image, label) and returns (augmented_image, label)
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
    
    def enhanced_augment_image(image, label):
        """Apply enhanced data augmentation to an image."""
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
                image_width_float = tf.cast(image_width, tf.float32)
                w_pixels = tf.cast(image_width_float * width_shift_range, tf.int32)
                w_shift = tf.random.uniform(
                    shape=[], minval=-w_pixels, maxval=w_pixels, dtype=tf.int32
                )
                image = tf.roll(image, shift=w_shift, axis=1)

            if height_shift_range > 0:
                image_height_float = tf.cast(image_height, tf.float32)
                h_pixels = tf.cast(image_height_float * height_shift_range, tf.int32)
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
    
    return enhanced_augment_image


def get_batch_augmentation_pipeline(
    config: Dict[str, Any] = None
) -> Callable:
    """Get a function that applies batch-level augmentations.
    
    Args:
        config: Configuration dictionary with augmentation parameters
        
    Returns:
        Function that takes (images, labels) and returns (augmented_images, augmented_labels)
    """
    if config is None:
        config = {}

    # Get augmentation parameters from config or use defaults
    apply_mixup = config.get("mixup", True)
    apply_cutmix = config.get("cutmix", True)
    mixup_alpha = config.get("mixup_alpha", 0.2)
    cutmix_alpha = config.get("cutmix_alpha", 1.0)

    def batch_augment(images, labels):
        """Apply batch-level augmentations like MixUp and CutMix"""
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
    
    return batch_augment


def get_validation_transforms(
    image_size: Tuple[int, int] = (224, 224)
) -> Callable:
    """Get a function that applies validation-time transformations.
    
    Args:
        image_size: Target size (height, width)
        
    Returns:
        Function that takes (image, label) and returns (processed_image, label)
    """
    def validation_transform(image, label):
        """Transforms for validation - center crop and normalization only"""
        # Resize to slightly larger than target size
        larger_size = (int(image_size[0] * 1.14), int(image_size[1] * 1.14))
        image = tf.image.resize(image, larger_size)

        # Center crop to target size
        image = tf.image.resize_with_crop_or_pad(image, image_size[0], image_size[1])

        # Ensure normalization
        image = tf.clip_by_value(image, 0.0, 1.0)

        return image, label
    
    return validation_transform```

---

### src/preprocessing/data_validator.py

```python
# src/preprocessing/data_validator.py
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
import numpy as np
import concurrent.futures
from tqdm.auto import tqdm
import tensorflow as tf

# Optional imports for image validation
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# Custom validation exception
class DataValidationError(Exception):
    """Exception raised for errors in data validation."""

    pass


class DataValidator:
    """
    Validates image datasets to ensure data quality before model training.

    This class performs various checks on image data:
    - Validates file existence and readability
    - Checks for corrupt images
    - Detects class imbalance
    - Verifies image dimensions and channel consistency
    - Identifies potential duplicate images
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the data validator.

        Args:
            config: Configuration dictionary with validation settings
        """
        self.config = config or {}
        self.validation_config = self.config.get("data_validation", {})

        # Validation settings with defaults
        self.min_samples_per_class = self.validation_config.get(
            "min_samples_per_class", 5
        )
        self.max_class_imbalance_ratio = self.validation_config.get(
            "max_class_imbalance_ratio", 10.0
        )
        self.min_image_dimensions = self.validation_config.get(
            "min_image_dimensions", (32, 32)
        )
        self.check_corrupt_images = self.validation_config.get(
            "check_corrupt_images", True
        )
        self.check_duplicates = self.validation_config.get(
            "check_duplicates", False
        )  # Expensive, off by default
        self.max_workers = self.validation_config.get("max_workers", 4)
        self.max_images_to_check = self.validation_config.get(
            "max_images_to_check", 1000
        )  # Limit for performance

        # Tracked validation issues
        self.validation_issues = {
            "corrupt_files": [],
            "small_dimensions": [],
            "wrong_channels": [],
            "potential_duplicates": [],
            "class_statistics": {},
            "warnings": [],
            "errors": [],
        }

        self.logger = logging.getLogger("DataValidator")

    def validate_dataset(self, data_dir: Union[str, Path]) -> Dict:
        """
        Perform comprehensive validation on a dataset directory.

        Args:
            data_dir: Path to the dataset directory (containing class subdirectories)

        Returns:
            Dictionary with validation results and statistics
        """
        data_dir = Path(data_dir)
        self.logger.info(f"Validating dataset in {data_dir}")

        # Reset validation issues
        self.validation_issues = {
            "corrupt_files": [],
            "small_dimensions": [],
            "wrong_channels": [],
            "potential_duplicates": [],
            "class_statistics": {},
            "warnings": [],
            "errors": [],
        }

        # Check if data directory exists
        if not data_dir.exists():
            error_msg = f"Data directory does not exist: {data_dir}"
            self.validation_issues["errors"].append(error_msg)
            self.logger.error(error_msg)
            return self.validation_issues

        # Get class directories
        class_dirs = [d for d in data_dir.iterdir() if d.is_dir()]

        if not class_dirs:
            error_msg = f"No class directories found in {data_dir}"
            self.validation_issues["errors"].append(error_msg)
            self.logger.error(error_msg)
            return self.validation_issues

        # Collect files by class
        class_files = {}
        for class_dir in class_dirs:
            class_name = class_dir.name
            image_files = self._get_image_files(class_dir)
            class_files[class_name] = image_files
            self.validation_issues["class_statistics"][class_name] = {
                "count": len(image_files),
                "valid_count": 0,
                "corrupt_count": 0,
                "dimensions": {},
            }

        # Check class balance
        self._check_class_balance(class_files)

        # Validate image files
        total_files = sum(len(files) for files in class_files.values())
        self.logger.info(
            f"Found {total_files} image files across {len(class_files)} classes"
        )

        # Limit the number of files to check if needed
        files_to_check = min(total_files, self.max_images_to_check)
        if files_to_check < total_files:
            self.logger.warning(
                f"Only validating {files_to_check} out of {total_files} images. "
                "Set max_images_to_check higher for more thorough validation."
            )

        # Flatten list of files with their class labels
        all_files_with_class = []
        for class_name, files in class_files.items():
            for file_path in files:
                all_files_with_class.append((file_path, class_name))

        # Randomly sample if we need to limit
        if files_to_check < total_files:
            indices = np.random.choice(
                len(all_files_with_class), size=files_to_check, replace=False
            )
            files_to_validate = [all_files_with_class[i] for i in indices]
        else:
            files_to_validate = all_files_with_class

        # Validate files in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(self._validate_image, file_path, class_name): (
                    file_path,
                    class_name,
                )
                for file_path, class_name in files_to_validate
            }

            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Validating images",
            ):
                file_path, class_name = futures[future]
                try:
                    result = future.result()
                    self._process_validation_result(result, file_path, class_name)
                except Exception as e:
                    self.logger.error(f"Error validating {file_path}: {str(e)}")
                    self.validation_issues["errors"].append(
                        f"Error validating {file_path}: {str(e)}"
                    )

        # Check for duplicate images (if enabled)
        if self.check_duplicates and total_files <= self.max_images_to_check:
            self._check_potential_duplicates(files_to_validate)

        # Generate summary
        self._generate_validation_summary()

        return self.validation_issues

    def _get_image_files(self, directory: Path) -> List[Path]:
        """Get all image files in a directory."""
        valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        return [
            f
            for f in directory.glob("**/*")
            if f.is_file() and f.suffix.lower() in valid_extensions
        ]

    def _check_class_balance(self, class_files: Dict[str, List[Path]]):
        """Check for class imbalance issues."""
        if not class_files:
            return

        # Get class counts
        class_counts = {
            class_name: len(files) for class_name, files in class_files.items()
        }
        total_files = sum(class_counts.values())
        avg_files_per_class = total_files / len(class_counts)

        # Check for small classes
        small_classes = []
        for class_name, count in class_counts.items():
            if count < self.min_samples_per_class:
                small_classes.append((class_name, count))
                self.validation_issues["warnings"].append(
                    f"Class '{class_name}' has only {count} samples (min: {self.min_samples_per_class})"
                )

        # Check for class imbalance
        if class_counts:
            max_count = max(class_counts.values())
            min_count = min(class_counts.values())
            imbalance_ratio = max_count / max(min_count, 1)

            largest_class = max(class_counts.items(), key=lambda x: x[1])[0]
            smallest_class = min(class_counts.items(), key=lambda x: x[1])[0]

            if imbalance_ratio > self.max_class_imbalance_ratio:
                self.validation_issues["warnings"].append(
                    f"Severe class imbalance detected: ratio {imbalance_ratio:.2f} "
                    f"(largest: '{largest_class}' with {max_count}, "
                    f"smallest: '{smallest_class}' with {min_count})"
                )
            elif imbalance_ratio > 2.0:
                self.validation_issues["warnings"].append(
                    f"Moderate class imbalance detected: ratio {imbalance_ratio:.2f} "
                    f"(largest: '{largest_class}' with {max_count}, "
                    f"smallest: '{smallest_class}' with {min_count})"
                )

    def _validate_image(self, file_path: Path, class_name: str) -> Dict:
        """
        Validate a single image file.

        Args:
            file_path: Path to the image file
            class_name: The class this image belongs to

        Returns:
            Dictionary with validation results
        """
        result = {
            "path": str(file_path),
            "class": class_name,
            "is_valid": True,
            "issues": [],
            "dimensions": None,
            "channels": None,
            "file_size_kb": None,
        }

        # Basic file checks
        if not file_path.exists():
            result["is_valid"] = False
            result["issues"].append("file_not_found")
            return result

        # Check file size
        try:
            file_size = file_path.stat().st_size
            result["file_size_kb"] = file_size / 1024

            if file_size == 0:
                result["is_valid"] = False
                result["issues"].append("empty_file")
                return result
        except Exception as e:
            result["is_valid"] = False
            result["issues"].append(f"file_stat_error: {str(e)}")
            return result

        # Approach 1: Use PIL for image validation if available
        if PIL_AVAILABLE and self.check_corrupt_images:
            try:
                with Image.open(file_path) as img:
                    # This verifies the file can be read
                    img.verify()

                # Need to reopen after verify
                with Image.open(file_path) as img:
                    result["dimensions"] = img.size
                    if len(img.size) < 2:
                        result["is_valid"] = False
                        result["issues"].append("invalid_dimensions")

                    # Check channels
                    if hasattr(img, "mode"):
                        if img.mode == "RGB":
                            result["channels"] = 3
                        elif img.mode == "RGBA":
                            result["channels"] = 4
                        elif img.mode == "L":
                            result["channels"] = 1
                        else:
                            result["channels"] = 0
                            result["issues"].append(f"unusual_color_mode: {img.mode}")

                    # Check dimensions
                    width, height = result["dimensions"]
                    min_width, min_height = self.min_image_dimensions
                    if width < min_width or height < min_height:
                        result["is_valid"] = True  # still valid but flag it
                        result["issues"].append("small_dimensions")
            except Exception as e:
                result["is_valid"] = False
                result["issues"].append(f"corrupt_image: {str(e)}")
                return result

        # Approach 2: Try loading with TensorFlow (less thorough but always available)
        if not PIL_AVAILABLE or not result.get("dimensions"):
            try:
                # Read the file
                image_data = tf.io.read_file(str(file_path))

                # Decode image
                try:
                    img = tf.io.decode_image(
                        image_data, channels=0, expand_animations=False
                    )

                    # Get dimensions
                    shape = img.shape
                    if len(shape) >= 2:
                        result["dimensions"] = (shape[1], shape[0])  # width, height
                        result["channels"] = shape[2] if len(shape) > 2 else 1

                        # Check dimensions
                        min_width, min_height = self.min_image_dimensions
                        if shape[1] < min_width or shape[0] < min_height:
                            result["is_valid"] = True  # still valid but flag it
                            result["issues"].append("small_dimensions")
                    else:
                        result["is_valid"] = False
                        result["issues"].append("invalid_dimensions")
                except Exception as e:
                    result["is_valid"] = False
                    result["issues"].append(f"corrupt_image: {str(e)}")
                    return result

            except Exception as e:
                result["is_valid"] = False
                result["issues"].append(f"tf_read_error: {str(e)}")
                return result

        return result

    def _process_validation_result(
        self, result: Dict, file_path: Path, class_name: str
    ):
        """Process and store validation results for an image."""
        class_stats = self.validation_issues["class_statistics"][class_name]

        if result["is_valid"]:
            class_stats["valid_count"] += 1

            # Track dimension statistics
            if result["dimensions"]:
                dim_key = f"{result['dimensions'][0]}x{result['dimensions'][1]}"
                if dim_key not in class_stats["dimensions"]:
                    class_stats["dimensions"][dim_key] = 0
                class_stats["dimensions"][dim_key] += 1
        else:
            class_stats["corrupt_count"] += 1
            self.validation_issues["corrupt_files"].append(str(file_path))

        # Track specific issues
        for issue in result["issues"]:
            if issue == "small_dimensions":
                self.validation_issues["small_dimensions"].append(str(file_path))
            elif issue == "unusual_color_mode" or issue.startswith("channel_mismatch"):
                self.validation_issues["wrong_channels"].append(str(file_path))

    def _check_potential_duplicates(self, files_to_validate: List[Tuple[Path, str]]):
        """
        Simple duplicate check based on file size.
        Note: This is a basic heuristic and won't catch all duplicates.
        For more accurate detection, consider using image hashing libraries.
        """
        # Group files by size
        files_by_size = {}
        for file_path, _ in files_to_validate:
            try:
                size = file_path.stat().st_size
                if size not in files_by_size:
                    files_by_size[size] = []
                files_by_size[size].append(str(file_path))
            except Exception:
                pass

        # Check for potential duplicates
        for size, files in files_by_size.items():
            if len(files) > 1:
                for file in files:
                    self.validation_issues["potential_duplicates"].append(file)

                if len(files) > 2:
                    self.validation_issues["warnings"].append(
                        f"Found {len(files)} files with identical size ({size} bytes), "
                        "suggesting potential duplicates"
                    )

    def _generate_validation_summary(self):
        """Generate a summary of validation results."""
        # Count total issues
        num_corrupt = len(self.validation_issues["corrupt_files"])
        num_small = len(self.validation_issues["small_dimensions"])
        num_wrong_channels = len(self.validation_issues["wrong_channels"])
        num_duplicates = (
            len(self.validation_issues["potential_duplicates"]) // 2
        )  # Count pairs

        class_stats = self.validation_issues["class_statistics"]
        total_files = sum(stats["count"] for stats in class_stats.values())
        valid_files = sum(stats["valid_count"] for stats in class_stats.values())

        # Add summary to warnings
        if num_corrupt > 0:
            percent = (num_corrupt / total_files) * 100
            self.validation_issues["warnings"].append(
                f"Found {num_corrupt} potentially corrupt files ({percent:.1f}% of dataset)"
            )

        if num_small > 0:
            percent = (num_small / total_files) * 100
            self.validation_issues["warnings"].append(
                f"Found {num_small} images with dimensions smaller than {self.min_image_dimensions} "
                f"({percent:.1f}% of dataset)"
            )

        if num_wrong_channels > 0:
            percent = (num_wrong_channels / total_files) * 100
            self.validation_issues["warnings"].append(
                f"Found {num_wrong_channels} images with unusual color channels "
                f"({percent:.1f}% of dataset)"
            )

        if num_duplicates > 0:
            percent = (num_duplicates * 2 / total_files) * 100
            self.validation_issues["warnings"].append(
                f"Found {num_duplicates} potential duplicate image pairs "
                f"({percent:.1f}% of dataset)"
            )

        # Add summary information
        self.validation_issues["summary"] = {
            "total_files": total_files,
            "valid_files": valid_files,
            "corrupt_files": num_corrupt,
            "small_dimension_files": num_small,
            "wrong_channel_files": num_wrong_channels,
            "potential_duplicate_pairs": num_duplicates,
            "num_classes": len(class_stats),
            "class_distribution": {
                name: stats["count"] for name, stats in class_stats.items()
            },
        }

    def get_validation_summary(self) -> Dict:
        """Get a summary of the validation results."""
        if "summary" not in self.validation_issues:
            self._generate_validation_summary()

        return {
            "summary": self.validation_issues["summary"],
            "warnings": self.validation_issues["warnings"],
            "errors": self.validation_issues["errors"],
        }

    def print_validation_report(self):
        """Print a formatted validation report to the console."""
        if "summary" not in self.validation_issues:
            self._generate_validation_summary()

        summary = self.validation_issues["summary"]

        print("\n" + "=" * 80)
        print("DATA VALIDATION REPORT")
        print("=" * 80)

        print(f"\nTotal files examined: {summary['total_files']}")
        print(
            f"Valid files: {summary['valid_files']} ({summary['valid_files']/max(1, summary['total_files'])*100:.1f}%)"
        )
        print(f"Number of classes: {summary['num_classes']}")

        # Print issues
        issues_found = False

        if summary["corrupt_files"] > 0:
            issues_found = True
            print(f"\nCorrupt files found: {summary['corrupt_files']}")
            if len(self.validation_issues["corrupt_files"]) > 0:
                print("First few corrupt files:")
                for path in self.validation_issues["corrupt_files"][:5]:
                    print(f"  - {path}")
                if len(self.validation_issues["corrupt_files"]) > 5:
                    print(
                        f"  ... and {len(self.validation_issues['corrupt_files']) - 5} more"
                    )

        if summary["small_dimension_files"] > 0:
            issues_found = True
            print(f"\nSmall dimension files found: {summary['small_dimension_files']}")
            print(
                f"Minimum dimensions threshold: {self.min_image_dimensions[0]}x{self.min_image_dimensions[1]}"
            )

        if summary["wrong_channel_files"] > 0:
            issues_found = True
            print(
                f"\nFiles with unusual color channels: {summary['wrong_channel_files']}"
            )

        if summary["potential_duplicate_pairs"] > 0:
            issues_found = True
            print(
                f"\nPotential duplicate pairs: {summary['potential_duplicate_pairs']}"
            )

        # Print class distribution
        print("\nClass distribution:")
        class_counts = summary["class_distribution"]
        max_count = max(class_counts.values()) if class_counts else 0
        for class_name, count in sorted(
            class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            bar = "█" * int(50 * count / max(1, max_count))
            print(f"  {class_name.ljust(30)} {str(count).rjust(5)} {bar}")

        # Print warnings and errors
        if self.validation_issues["warnings"]:
            print("\nWarnings:")
            for warning in self.validation_issues["warnings"]:
                print(f"  - {warning}")

        if self.validation_issues["errors"]:
            print("\nErrors:")
            for error in self.validation_issues["errors"]:
                print(f"  - {error}")

        if (
            not issues_found
            and not self.validation_issues["warnings"]
            and not self.validation_issues["errors"]
        ):
            print("\n✓ No issues found. Dataset looks good!")

        print("\n" + "=" * 80)


def validate_dataset(data_dir: Union[str, Path], config: Dict = None) -> Dict:
    """
    Convenience function to validate a dataset.

    Args:
        data_dir: Path to the dataset directory
        config: Configuration dictionary

    Returns:
        Validation results
    """
    validator = DataValidator(config)
    results = validator.validate_dataset(data_dir)
    validator.print_validation_report()
    return results
```

---

### src/preprocessing/dataset_loader.py

```python
"""
Dataset loader module for handling dataset loading operations.
This separates concerns from the data_loader.py file to focus only on loading operations.
"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Any, Optional, Union

from src.config.config import get_paths
from src.preprocessing.data_validator import DataValidator
from src.utils.seed_utils import set_global_seeds


class DatasetLoader:
    """Handles dataset loading operations from files and directories."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the dataset loader with given configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = get_paths()
        
        # Set random seed for reproducibility
        self.seed = self.config.get("seed", 42)
        set_global_seeds(self.seed)
        
        # Get validation configuration
        validation_config = config.get("data_validation", {})
        self.validate_data = validation_config.get("enabled", True)
        self.validator = DataValidator(config)
        
        # Set image parameters
        self.image_size = self.config.get("data", {}).get("image_size", (224, 224))
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)

    def load_dataset_from_directory(
        self, 
        data_dir: Union[str, Path], 
        validation_split: float = 0.2,
        test_split: float = 0.1,
        use_saved_splits: bool = None
    ) -> Tuple[List[str], List[int], Dict[int, str]]:
        """Load dataset files and labels from directory structure.
        
        Args:
            data_dir: Path to data directory
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            use_saved_splits: Whether to try loading from saved splits
            
        Returns:
            Tuple of (file_paths, labels, class_names)
            
        Raises:
            ValueError: If dataset validation fails or no class directories found
        """
        # Determine if we should use saved splits
        if use_saved_splits is None:
            use_saved_splits = self.config.get("data", {}).get("use_saved_splits", False)

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
                return self._load_from_saved_splits(splits_dir)
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
            
        return all_files, all_labels, class_names
    
    def _load_from_saved_splits(self, splits_dir: Union[str, Path]) -> Tuple[List[str], List[int], Dict[int, str]]:
        """Load dataset files and labels from saved splits.
        
        Args:
            splits_dir: Path to directory containing splits
            
        Returns:
            Tuple of (file_paths, labels, class_names)
            
        Raises:
            ValueError: If saved splits cannot be loaded
        """
        # Load metadata if available
        metadata_path = Path(splits_dir) / "splits_metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Splits metadata file not found at {metadata_path}")
            
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print(f"Loaded splits metadata from {metadata_path}")
        
        # Load class mapping
        class_mapping_path = Path(splits_dir) / "class_mapping.json"
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
        
        # Load files and labels from the splits
        all_files = []
        all_labels = []
        
        # Try to load from train split
        train_split_path = Path(splits_dir) / "train_split.csv"
        if train_split_path.exists():
            df = pd.read_csv(train_split_path)
            if "file_path" in df.columns and "label" in df.columns:
                valid_paths = []
                valid_labels = []
                for _, row in df.iterrows():
                    if os.path.exists(row["file_path"]):
                        valid_paths.append(row["file_path"])
                        valid_labels.append(row["label"])
                
                all_files.extend(valid_paths)
                all_labels.extend(valid_labels)
                print(f"Loaded {len(valid_paths)} files from train split")
        
        # Try to load from validation split
        val_split_path = Path(splits_dir) / "val_split.csv"
        if val_split_path.exists():
            df = pd.read_csv(val_split_path)
            if "file_path" in df.columns and "label" in df.columns:
                valid_paths = []
                valid_labels = []
                for _, row in df.iterrows():
                    if os.path.exists(row["file_path"]):
                        valid_paths.append(row["file_path"])
                        valid_labels.append(row["label"])
                
                all_files.extend(valid_paths)
                all_labels.extend(valid_labels)
                print(f"Loaded {len(valid_paths)} files from validation split")
        
        # Try to load from test split
        test_split_path = Path(splits_dir) / "test_split.csv"
        if test_split_path.exists():
            df = pd.read_csv(test_split_path)
            if "file_path" in df.columns and "label" in df.columns:
                valid_paths = []
                valid_labels = []
                for _, row in df.iterrows():
                    if os.path.exists(row["file_path"]):
                        valid_paths.append(row["file_path"])
                        valid_labels.append(row["label"])
                
                all_files.extend(valid_paths)
                all_labels.extend(valid_labels)
                print(f"Loaded {len(valid_paths)} files from test split")
        
        if not all_files:
            raise ValueError("No valid files found in saved splits")
            
        return all_files, all_labels, class_names
        
    def save_dataset_splits(
        self,
        file_paths: List[str],
        labels: List[int],
        class_names: Dict[int, str],
        indices: Dict[str, List[int]],
        output_dir: Union[str, Path]
    ) -> Dict[str, str]:
        """Save dataset splits to disk for reproducibility.
        
        Args:
            file_paths: List of file paths
            labels: List of labels
            class_names: Dictionary mapping indices to class names
            indices: Dictionary mapping split names to list of indices
            output_dir: Directory to save splits
            
        Returns:
            Dictionary mapping split names to file paths
        """
        output_path = Path(output_dir)
        splits_dir = output_path / "splits"
        os.makedirs(splits_dir, exist_ok=True)

        # Save class mapping
        class_to_idx = {name: idx for idx, name in class_names.items()}
        idx_to_class = class_names

        class_mapping_path = splits_dir / "class_mapping.json"
        with open(class_mapping_path, "w") as f:
            json.dump(
                {"class_to_idx": class_to_idx, "idx_to_class": idx_to_class},
                f,
                indent=2,
            )

        print(f"Class mapping saved to {class_mapping_path}")

        split_paths = {}

        # For each split, save the corresponding files and labels
        for split_name, split_indices in indices.items():
            split_files = [file_paths[i] for i in split_indices]
            split_labels = [labels[i] for i in split_indices]
            
            # Create DataFrame with the split data
            split_df = pd.DataFrame({
                "file_path": split_files,
                "label": split_labels,
                "class_name": [class_names[label] for label in split_labels]
            })
            
            # Save to CSV
            split_path = splits_dir / f"{split_name}_split.csv"
            split_df.to_csv(split_path, index=False)
            split_paths[split_name] = str(split_path)
            
            print(f"{split_name.capitalize()} split saved to {split_path} ({len(split_df)} samples)")

        # Save metadata about the splits
        train_split = indices.get("train", [])
        val_split = indices.get("val", [])
        test_split = indices.get("test", [])
        
        metadata = {
            "splits": {
                k: {"path": v, "size": len(pd.read_csv(v))}
                for k, v in split_paths.items()
            },
            "class_mapping_path": str(class_mapping_path),
            "num_classes": len(class_names),
            "image_size": list(self.image_size),
            "creation_timestamp": str(pd.Timestamp.now()),
            "split_percentages": {
                "train": len(train_split) / len(file_paths) if file_paths else 0,
                "validation": len(val_split) / len(file_paths) if file_paths else 0,
                "test": len(test_split) / len(file_paths) if file_paths else 0,
            },
        }

        metadata_path = splits_dir / "splits_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset splits metadata saved to {metadata_path}")
        return split_paths
        
    def split_dataset(
        self, 
        file_paths: List[str], 
        labels: List[int],
        validation_split: float = 0.2,
        test_split: float = 0.1
    ) -> Dict[str, List[int]]:
        """Split dataset into training, validation, and test sets.
        
        Args:
            file_paths: List of file paths
            labels: List of labels
            validation_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            
        Returns:
            Dictionary mapping split names to list of indices
        """
        # Create a shuffled dataset index
        dataset_size = len(file_paths)
        indices = np.arange(dataset_size)
        np.random.seed(self.seed)
        np.random.shuffle(indices)
        
        # Calculate split sizes
        train_size = int(dataset_size * (1 - validation_split - test_split))
        val_size = int(dataset_size * validation_split)
        test_size = dataset_size - train_size - val_size
        
        # Create split indices
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        print(f"Splitting dataset:")
        print(f"  - Training: {train_size} images ({(train_size/dataset_size)*100:.1f}%)")
        print(f"  - Validation: {val_size} images ({(val_size/dataset_size)*100:.1f}%)")
        print(f"  - Test: {test_size} images ({(test_size/dataset_size)*100:.1f}%)")
        
        return {
            "train": train_indices.tolist(),
            "val": val_indices.tolist(),
            "test": test_indices.tolist()
        }
        
    def get_class_weights(self, labels: List[int], indices: List[int]) -> Dict[int, float]:
        """Calculate class weights for imbalanced datasets.
        
        Args:
            labels: List of all labels
            indices: List of indices to consider (e.g., training indices)
            
        Returns:
            Dictionary mapping class indices to weights
        """
        # Extract the labels for the given indices
        subset_labels = [labels[i] for i in indices]
        
        # Count classes
        class_counts = {}
        for label in subset_labels:
            if label not in class_counts:
                class_counts[label] = 0
            class_counts[label] += 1
        
        # Calculate weights (inversely proportional to frequency)
        total = len(subset_labels)
        n_classes = len(class_counts)
        
        # Use balanced formula: total_samples / (n_classes * class_count)
        weights = {}
        for label, count in class_counts.items():
            weights[label] = total / (n_classes * count) if count > 0 else 1.0
        
        return weights```

---

### src/preprocessing/dataset_pipeline.py

```python
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

        # Get batch and performance settings
        training_config = config.get("training", {})
        self.batch_size = training_config.get("batch_size", 32)

        # Get hardware configuration for performance tuning
        hardware_config = config.get("hardware", {})
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
```

---

### src/scripts/__init__.py

```python
```

---

### src/scripts/evaluate.py

```python
#!/usr/bin/env python3
"""
Evaluate a trained model on a dataset
"""

import os
import argparse

import tensorflow as tf
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm


from src.config.config import get_paths
from src.config.config_loader import ConfigLoader
from src.preprocessing.data_loader import DataLoader
from src.evaluation.metrics import evaluate_model
from src.evaluation.visualization import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_class_distribution,
    plot_misclassified_examples,
)
from src.utils.report_generator import ReportGenerator


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model (.h5 file)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to the dataset directory for evaluation",
    )
    parser.add_argument(
        "--config", type=str, default=None, help="Path to the configuration file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for evaluation (overrides config)",
    )
    parser.add_argument(
        "--visualize_misclassified",
        action="store_true",
        help="Generate visualizations of misclassified samples",
    )
    args = parser.parse_args()

    # Get project paths
    paths = get_paths()

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()

    # Override batch size if provided
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # If model path is inside trials directory, use its evaluation folder
        model_path = Path(args.model_path)
        model_dir = model_path.parent

        if "trials" in str(model_dir):
            output_dir = model_dir / "evaluation"
        else:
            # Default to a timestamp-based directory
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = paths.trials_dir / "evaluations" / f"eval_{timestamp}"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluation results will be saved to: {output_dir}")

    # Load model
    try:
        print(f"Loading model from {args.model_path}...")
        model = tf.keras.models.load_model(args.model_path)
        print(f"Model loaded successfully.")

        # Print model summary
        model.summary()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load data
    data_loader = DataLoader(config)
    _, _, test_data, class_names = data_loader.load_data(args.data_dir)

    if test_data is None:
        print("Error: No test data available for evaluation")
        return

    print(f"Evaluating model on {len(class_names)} classes")

    # Evaluate model
    print("Evaluating model...")
    metrics_path = output_dir / "metrics.json"
    metrics = evaluate_model(
        model,
        test_data,
        class_names=class_names,
        metrics_path=metrics_path,
        use_tqdm=True,
    )

    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
    print(f"  Loss: {metrics.get('loss', 0):.4f}")

    if "f1_macro" in metrics:
        print(f"  F1 Score (Macro): {metrics.get('f1_macro', 0):.4f}")

    if "precision_macro" in metrics:
        print(f"  Precision (Macro): {metrics.get('precision_macro', 0):.4f}")

    if "recall_macro" in metrics:
        print(f"  Recall (Macro): {metrics.get('recall_macro', 0):.4f}")

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Get predictions for visualization
    print("Generating predictions for visualization...")
    all_x = []
    all_y_true = []
    all_y_pred_prob = []

    # Use tqdm for progress tracking
    for batch_idx, (x, y) in enumerate(tqdm(test_data, desc="Predicting")):
        # Get predictions for this batch
        y_pred = model.predict(x, verbose=0)
        all_y_pred_prob.append(y_pred)
        all_y_true.append(y)

        # For misclassified visualization, save the images too
        if args.visualize_misclassified:
            all_x.append(x)

    # Concatenate all batches
    y_pred_prob = np.vstack(all_y_pred_prob)
    y_true = np.vstack(all_y_true)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # For misclassified visualization, concatenate the images
    if args.visualize_misclassified:
        x_test = np.vstack(all_x)

    # Plot confusion matrix
    print("Generating confusion matrix...")
    cm_path = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)
    print(f"Confusion matrix saved to {cm_path}")

    # Plot normalized confusion matrix
    cm_norm_path = plots_dir / "confusion_matrix_normalized.png"
    plot_confusion_matrix(
        y_true, y_pred, class_names, save_path=cm_norm_path, normalize=True
    )
    print(f"Normalized confusion matrix saved to {cm_norm_path}")

    # Plot ROC curve
    print("Generating ROC curve...")
    roc_path = plots_dir / "roc_curve.png"
    plot_roc_curve(y_true, y_pred_prob, class_names, save_path=roc_path)
    print(f"ROC curve saved to {roc_path}")

    # Plot precision-recall curve
    print("Generating precision-recall curve...")
    pr_path = plots_dir / "precision_recall_curve.png"
    plot_precision_recall_curve(y_true, y_pred_prob, class_names, save_path=pr_path)
    print(f"Precision-recall curve saved to {pr_path}")

    # Plot class distribution
    print("Generating class distribution...")
    dist_path = plots_dir / "class_distribution.png"
    plot_class_distribution(y_true, class_names, save_path=dist_path)
    print(f"Class distribution saved to {dist_path}")

    # Plot misclassified examples if requested
    if args.visualize_misclassified:
        print("Generating misclassified examples visualization...")
        misclass_path = plots_dir / "misclassified_examples.png"
        plot_misclassified_examples(
            x_test,
            y_true,
            y_pred,
            class_names,
            num_examples=25,
            save_path=misclass_path,
        )
        print(f"Misclassified examples saved to {misclass_path}")

    # Generate HTML report
    print("Generating evaluation report...")
    report_generator = ReportGenerator(config)
    report_context = {
        "model_path": args.model_path,
        "metrics": metrics,
        "plots": {
            "confusion_matrix": str(cm_path),
            "confusion_matrix_normalized": str(cm_norm_path),
            "roc_curve": str(roc_path),
            "precision_recall_curve": str(pr_path),
            "class_distribution": str(dist_path),
        },
    }

    if args.visualize_misclassified:
        report_context["plots"]["misclassified_examples"] = str(misclass_path)

    report_path = output_dir / "evaluation_report.html"

    # Create a simple report template
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2c3e50; }}
            .metrics {{ margin: 20px 0; }}
            .metrics table {{ border-collapse: collapse; width: 100%; }}
            .metrics th, .metrics td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics th {{ background-color: #f2f2f2; }}
            .plots {{ display: flex; flex-wrap: wrap; justify-content: center; }}
            .plot {{ margin: 10px; text-align: center; }}
            .plot img {{ max-width: 800px; border: 1px solid #ddd; }}
        </style>
    </head>
    <body>
        <h1>Model Evaluation Report</h1>
        <p>Model: {os.path.basename(args.model_path)}</p>
        
        <h2>Metrics</h2>
        <div class="metrics">
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Accuracy</td><td>{metrics.get('accuracy', 0):.4f}</td></tr>
                <tr><td>Loss</td><td>{metrics.get('loss', 0):.4f}</td></tr>
                <tr><td>F1 Score (Macro)</td><td>{metrics.get('f1_macro', 0):.4f}</td></tr>
                <tr><td>Precision (Macro)</td><td>{metrics.get('precision_macro', 0):.4f}</td></tr>
                <tr><td>Recall (Macro)</td><td>{metrics.get('recall_macro', 0):.4f}</td></tr>
            </table>
        </div>
        
        <h2>Visualizations</h2>
        <div class="plots">
            <div class="plot">
                <h3>Confusion Matrix</h3>
                <img src="{os.path.relpath(cm_path, output_dir)}" alt="Confusion Matrix">
            </div>
            
            <div class="plot">
                <h3>Normalized Confusion Matrix</h3>
                <img src="{os.path.relpath(cm_norm_path, output_dir)}" alt="Normalized Confusion Matrix">
            </div>
            
            <div class="plot">
                <h3>ROC Curve</h3>
                <img src="{os.path.relpath(roc_path, output_dir)}" alt="ROC Curve">
            </div>
            
            <div class="plot">
                <h3>Precision-Recall Curve</h3>
                <img src="{os.path.relpath(pr_path, output_dir)}" alt="Precision-Recall Curve">
            </div>
            
            <div class="plot">
                <h3>Class Distribution</h3>
                <img src="{os.path.relpath(dist_path, output_dir)}" alt="Class Distribution">
            </div>
            
            {f'<div class="plot"><h3>Misclassified Examples</h3><img src="{os.path.relpath(misclass_path, output_dir)}" alt="Misclassified Examples"></div>' if args.visualize_misclassified else ''}
        </div>
    </body>
    </html>
    """

    with open(report_path, "w") as f:
        f.write(report_html)

    print(f"Evaluation completed. Report generated at {report_path}")


if __name__ == "__main__":
    main()
```

---

### src/scripts/preprocess_data.py

```python
#!/usr/bin/env python3
"""
Script to preprocess raw image data for the plant disease detection model.
This script organizes and preprocesses images from the raw data directory
into the processed data directory with the expected structure.
"""

import os
import shutil
import argparse
from pathlib import Path
import concurrent.futures
from tqdm import tqdm
import tensorflow as tf
import numpy as np

# Configure TensorFlow to use CPU (if needed)
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def preprocess_image(src_path, dst_path, target_size=(224, 224)):
    """Preprocess a single image and save it to the destination path.
    
    Args:
        src_path: Source image path
        dst_path: Destination image path
        target_size: Target size for resizing (height, width)
    """
    try:
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        
        # If we just want to copy the file without processing
        if target_size is None:
            shutil.copy2(src_path, dst_path)
            return True
        
        # Load and preprocess the image
        img = tf.io.read_file(str(src_path))
        
        # Decode based on extension
        if src_path.lower().endswith(('.jpg', '.jpeg')):
            img = tf.image.decode_jpeg(img, channels=3)
        elif src_path.lower().endswith('.png'):
            img = tf.image.decode_png(img, channels=3)
        else:
            img = tf.image.decode_image(img, channels=3, expand_animations=False)
        
        # Resize to target size
        img = tf.image.resize(img, target_size)
        
        # Ensure image is in 0-255 range and uint8 format
        img = tf.clip_by_value(img, 0, 255)
        img = tf.cast(img, tf.uint8)
        
        # Save processed image
        img_encoded = tf.image.encode_jpeg(img, quality=95)
        tf.io.write_file(str(dst_path), img_encoded)
        
        return True
    except Exception as e:
        print(f"Error processing {src_path}: {e}")
        return False


def preprocess_dataset(raw_dir, processed_dir, target_size=(224, 224), num_workers=4, copy_only=False):
    """Preprocess all images in the raw directory and save to processed directory.
    
    Args:
        raw_dir: Source directory with raw images
        processed_dir: Destination directory for processed images
        target_size: Target size for resizing (height, width)
        num_workers: Number of parallel workers
        copy_only: If True, just copy files without processing
    """
    raw_dir = Path(raw_dir)
    processed_dir = Path(processed_dir)
    
    # Ensure processed directory exists
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all class directories
    class_dirs = [d for d in raw_dir.iterdir() if d.is_dir()]
    
    if not class_dirs:
        print(f"No class directories found in {raw_dir}")
        # Check if there are image files directly in the raw directory
        image_files = list(raw_dir.glob("*.jpg")) + list(raw_dir.glob("*.jpeg")) + list(raw_dir.glob("*.png"))
        if image_files:
            print(f"Found {len(image_files)} images directly in {raw_dir}. These should be organized into class directories.")
        return
    
    print(f"Found {len(class_dirs)} class directories")
    
    # Initialize counters
    total_images = 0
    processed_images = 0
    failed_images = 0
    
    # Process each class directory
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"Processing class: {class_name}")
        
        # Create corresponding directory in processed_dir
        dest_class_dir = processed_dir / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all image files in this class
        image_files = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_files.extend(list(class_dir.glob(ext)))
        
        if not image_files:
            print(f"  No images found in {class_dir}")
            continue
        
        print(f"  Found {len(image_files)} images")
        total_images += len(image_files)
        
        # Prepare preprocessing tasks
        tasks = []
        for src_path in image_files:
            # Destination filename - keep the same as source
            dst_filename = src_path.name
            dst_path = dest_class_dir / dst_filename
            
            # Add this task
            tasks.append((str(src_path), str(dst_path)))
        
        # Process images in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # If copy_only is True, set target_size to None
            size = None if copy_only else target_size
            
            futures = {
                executor.submit(preprocess_image, src, dst, size): (src, dst)
                for src, dst in tasks
            }
            
            # Track progress
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks), desc=f"Processing {class_name}"):
                src, dst = futures[future]
                try:
                    success = future.result()
                    if success:
                        processed_images += 1
                    else:
                        failed_images += 1
                except Exception as e:
                    print(f"Error processing {src}: {e}")
                    failed_images += 1
    
    # Print summary
    print("\nPreprocessing complete!")
    print(f"Total images: {total_images}")
    print(f"Successfully processed: {processed_images}")
    print(f"Failed: {failed_images}")
    
    if processed_images > 0:
        success_rate = (processed_images / total_images) * 100
        print(f"Success rate: {success_rate:.2f}%")
    
    print(f"\nProcessed data saved to: {processed_dir}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess plant disease dataset")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Directory with raw images")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Directory for processed images")
    parser.add_argument("--height", type=int, default=224, help="Target height for resizing")
    parser.add_argument("--width", type=int, default=224, help="Target width for resizing")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--copy_only", action="store_true", help="Just copy files without processing")
    
    args = parser.parse_args()
    
    # Make paths absolute if they're relative
    base_dir = Path.cwd()
    raw_dir = Path(args.raw_dir)
    if not raw_dir.is_absolute():
        raw_dir = base_dir / raw_dir
    
    processed_dir = Path(args.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = base_dir / processed_dir
    
    # Run preprocessing
    print(f"Processing images from {raw_dir} to {processed_dir}")
    print(f"Target size: {args.height}x{args.width}")
    
    preprocess_dataset(
        raw_dir,
        processed_dir,
        target_size=(args.height, args.width),
        num_workers=args.workers,
        copy_only=args.copy_only
    )


if __name__ == "__main__":
    main()```

---

### src/scripts/registry_cli.py

```python
#!/usr/bin/env python3
"""
Command line interface for the model registry.
This script allows you to manage the model registry from the command line.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
from tabulate import tabulate

# Add parent directory to path to allow imports
parent_dir = str(Path(__file__).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.model_registry.registry_manager import ModelRegistryManager


def list_models(registry, args):
    """List all models in the registry"""
    models = registry.list_models()

    if not models:
        print("No models found in registry")
        return

    # Collect details for each model
    model_details = []
    for model_name in models:
        model_info = registry._registry["models"][model_name]
        best_run_id = model_info.get("best_run")

        if best_run_id:
            best_run = model_info["runs"][best_run_id]
            accuracy = best_run.get("metrics", {}).get("test_accuracy", 0)
            loss = best_run.get("metrics", {}).get("test_loss", 0)
        else:
            accuracy = 0
            loss = 0

        model_details.append(
            {
                "Model": model_name,
                "Total Runs": model_info.get("total_runs", 0),
                "Best Run": best_run_id,
                "Best Accuracy": f"{accuracy:.4f}",
                "Best Loss": f"{loss:.4f}",
            }
        )

    # Create DataFrame and display as table
    df = pd.DataFrame(model_details)
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


def list_runs(registry, args):
    """List all runs for a specific model"""
    model_name = args.model

    if model_name not in registry._registry["models"]:
        print(f"Model {model_name} not found in registry")
        return

    runs = registry.list_runs(model_name)
    model_info = registry._registry["models"][model_name]
    best_run_id = model_info.get("best_run")

    if not runs:
        print(f"No runs found for model {model_name}")
        return

    # Collect details for each run
    run_details = []
    for run_id in runs:
        run_info = model_info["runs"][run_id]
        accuracy = run_info.get("metrics", {}).get("test_accuracy", 0)
        loss = run_info.get("metrics", {}).get("test_loss", 0)
        timestamp = run_info.get("timestamp", "")
        status = run_info.get("status", "unknown")
        is_best = run_id == best_run_id

        run_details.append(
            {
                "Run ID": run_id,
                "Timestamp": timestamp,
                "Accuracy": f"{accuracy:.4f}",
                "Loss": f"{loss:.4f}",
                "Status": status,
                "Best": "✓" if is_best else "",
            }
        )

    # Sort by timestamp (newest first)
    run_details.sort(key=lambda x: x["Timestamp"], reverse=True)

    # Create DataFrame and display as table
    df = pd.DataFrame(run_details)
    print(f"\nRuns for model: {model_name}")
    print(f"Total runs: {len(runs)}")
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False))


def show_details(registry, args):
    """Show detailed information about a model or run"""
    model_name = args.model
    run_id = args.run

    if model_name not in registry._registry["models"]:
        print(f"Model {model_name} not found in registry")
        return

    model_info = registry._registry["models"][model_name]

    if run_id:
        # Show run details
        if run_id not in model_info["runs"]:
            print(f"Run {run_id} not found for model {model_name}")
            return

        run_info = model_info["runs"][run_id]
        print(f"\nDetails for {model_name} run {run_id}:")
        print(f"  Path: {run_info.get('path')}")
        print(f"  Timestamp: {run_info.get('timestamp')}")
        print(f"  Status: {run_info.get('status')}")
        print(f"  Model file: {run_info.get('model_path')}")
        print(f"  Has checkpoints: {run_info.get('has_checkpoints')}")
        print(f"  Has TensorBoard logs: {run_info.get('has_tensorboard')}")

        print("\nMetrics:")
        metrics = run_info.get("metrics", {})
        for key, value in metrics.items():
            # Format the value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
            print(f"  {key}: {formatted_value}")
    else:
        # Show model details
        print(f"\nDetails for model {model_name}:")
        print(f"  Total runs: {model_info.get('total_runs')}")
        print(f"  Best run: {model_info.get('best_run')}")
        print(f"  Last run: {model_info.get('last_run')}")

        if model_info.get("best_run"):
            best_run = model_info["runs"][model_info.get("best_run")]
            print("\nBest run metrics:")
            metrics = best_run.get("metrics", {})
            for key, value in metrics.items():
                if key.startswith(("test_", "val_")) and isinstance(
                    value, (int, float)
                ):
                    print(f"  {key}: {value:.6f}")


def scan_trials(registry, args):
    """Scan trials directory for new models and runs"""
    new_runs = registry.scan_trials(rescan=args.rescan)
    if new_runs > 0:
        print(f"Added {new_runs} new runs to registry")
    else:
        print("No new runs found")


def compare_models(registry, args):
    """Compare multiple models"""
    if args.models:
        model_names = args.models
    else:
        # Use top N models if no names provided
        top_models = registry.get_best_models(top_n=args.top)
        model_names = [model["name"] for model in top_models]

    if not model_names:
        print("No models to compare")
        return

    print(f"Comparing models: {', '.join(model_names)}")

    # Get metrics to compare
    if args.metrics:
        metrics = args.metrics
    else:
        metrics = ["test_accuracy", "test_loss", "training_time"]

    # Compare models
    comparison_df = registry.compare_models(
        model_names=model_names, metrics=metrics, plot=True, output_dir=args.output_dir
    )

    if comparison_df.empty:
        print("No data available for comparison")
        return

    # Display comparison table
    print("\nModel Comparison:")
    print(tabulate(comparison_df, headers="keys", tablefmt="pretty", showindex=False))

    # Print path to generated plots
    if args.output_dir:
        print(f"\nComparison plots saved to: {args.output_dir}")
    else:
        print(
            f"\nComparison plots saved to: {registry.paths.trials_dir / 'comparisons'}"
        )


def export_registry(registry, args):
    """Export the registry to a file"""
    output_path = args.output
    path = registry.export_registry(output_path)
    print(f"Registry exported to {path}")


def import_registry(registry, args):
    """Import a registry from a file"""
    input_path = args.input
    success = registry.import_registry(input_path, merge=not args.replace)
    if success:
        print(f"Registry imported from {input_path}")
    else:
        print(f"Failed to import registry from {input_path}")


def generate_report(registry, args):
    """Generate an HTML report of the registry"""
    output_path = args.output
    path = registry.generate_registry_report(output_path)
    print(f"Registry report generated at {path}")


def delete_run(registry, args):
    """Delete a run from the registry"""
    model_name = args.model
    run_id = args.run

    if not args.force:
        confirmation = input(
            f"Are you sure you want to delete run {run_id} of model {model_name}? (y/n): "
        )
        if confirmation.lower() != "y":
            print("Deletion cancelled")
            return

    success = registry.delete_run(model_name, run_id, delete_files=args.delete_files)
    if success:
        print(f"Run {run_id} of model {model_name} deleted from registry")
        if args.delete_files:
            print("Run files were also deleted from disk")
    else:
        print(f"Failed to delete run {run_id} of model {model_name}")


def main():
    # Create the main parser
    parser = argparse.ArgumentParser(
        description="Model Registry CLI - Manage trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  registry_cli.py list                            # List all models
  registry_cli.py runs --model ResNet50           # List all runs for ResNet50
  registry_cli.py details --model ResNet50        # Show details for ResNet50
  registry_cli.py details --model ResNet50 --run run_20250304_123456_001  # Show run details
  registry_cli.py scan                            # Scan for new models and runs
  registry_cli.py compare --models ResNet50 MobileNetV2  # Compare models
  registry_cli.py report                          # Generate HTML report
        """,
    )

    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List models command
    list_parser = subparsers.add_parser("list", help="List all models in the registry")

    # List runs command
    runs_parser = subparsers.add_parser(
        "runs", help="List all runs for a specific model"
    )
    runs_parser.add_argument("--model", required=True, help="Name of the model")

    # Show details command
    details_parser = subparsers.add_parser(
        "details", help="Show detailed information about a model or run"
    )
    details_parser.add_argument("--model", required=True, help="Name of the model")
    details_parser.add_argument("--run", help="ID of the run (optional)")

    # Scan trials command
    scan_parser = subparsers.add_parser(
        "scan", help="Scan trials directory for new models and runs"
    )
    scan_parser.add_argument(
        "--rescan",
        action="store_true",
        help="Rescan all trials, even if already in registry",
    )

    # Compare models command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument(
        "--models", nargs="+", help="Names of models to compare"
    )
    compare_parser.add_argument("--metrics", nargs="+", help="Metrics to compare")
    compare_parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of top models to compare if no models specified",
    )
    compare_parser.add_argument(
        "--output-dir", help="Directory to save comparison plots"
    )

    # Export registry command
    export_parser = subparsers.add_parser(
        "export", help="Export the registry to a file"
    )
    export_parser.add_argument("--output", help="Path to export the registry")

    # Import registry command
    import_parser = subparsers.add_parser(
        "import", help="Import a registry from a file"
    )
    import_parser.add_argument(
        "--input", required=True, help="Path to the registry file"
    )
    import_parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing registry instead of merging",
    )

    # Generate report command
    report_parser = subparsers.add_parser(
        "report", help="Generate an HTML report of the registry"
    )
    report_parser.add_argument("--output", help="Path to save the report")

    # Delete run command
    delete_parser = subparsers.add_parser(
        "delete", help="Delete a run from the registry"
    )
    delete_parser.add_argument("--model", required=True, help="Name of the model")
    delete_parser.add_argument("--run", required=True, help="ID of the run")
    delete_parser.add_argument(
        "--delete-files", action="store_true", help="Also delete run files from disk"
    )
    delete_parser.add_argument(
        "--force", action="store_true", help="Skip confirmation prompt"
    )

    # Parse arguments
    args = parser.parse_args()

    # Check if a command was provided
    if not args.command:
        parser.print_help()
        return

    # Initialize registry manager
    registry = ModelRegistryManager()

    # Execute the appropriate command
    commands = {
        "list": list_models,
        "runs": list_runs,
        "details": show_details,
        "scan": scan_trials,
        "compare": compare_models,
        "export": export_registry,
        "import": import_registry,
        "report": generate_report,
        "delete": delete_run,
    }

    if args.command in commands:
        commands[args.command](registry, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

---

### src/scripts/train.py

```python
#!/usr/bin/env python3
"""
Train a model on a dataset with enhanced features
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras import backend as K
from pathlib import Path
import time

from src.config.config import get_paths
from src.config.config_loader import ConfigLoader
from src.preprocessing.data_loader import DataLoader
from src.models.enhanced_model_factory import EnhancedModelFactory
from src.training.trainer import Trainer
from src.training.lr_finder import (
    find_optimal_learning_rate,
    LearningRateFinderCallback,
)
from src.utils.seed_utils import set_global_seeds
from src.utils.hardware_utils import configure_hardware, print_hardware_summary
from src.model_registry.registry_manager import ModelRegistryManager


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model with enhanced features")
    parser.add_argument(
        "--model", type=str, required=True, help="Model architecture to train"
    )
    parser.add_argument(
        "--data_dir", type=str, default=None, help="Path to dataset directory"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config file")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override learning rate"
    )
    parser.add_argument(
        "--find_lr", action="store_true", help="Run learning rate finder"
    )
    parser.add_argument(
        "--attention", type=str, default=None, help="Attention type (se, cbam, spatial)"
    )
    parser.add_argument(
        "--use_enhanced",
        action="store_true",
        help="Use enhanced model variants like ResNet50_CBAM",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--hardware_summary",
        action="store_true",
        help="Print hardware configuration summary and exit",
    )
    args = parser.parse_args()

    # Print hardware summary if requested
    if args.hardware_summary:
        print_hardware_summary()
        return

    # Get project paths
    paths = get_paths()

    # Load configuration
    config_loader = ConfigLoader(args.config)
    config = config_loader.get_config()

    # Override config with command-line arguments
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.find_lr:
        if "lr_finder" not in config.get("training", {}):
            config["training"]["lr_finder"] = {}
        config["training"]["lr_finder"]["enabled"] = True
    if args.attention:
        config["training"]["attention_type"] = args.attention
    if args.seed:
        config["seed"] = args.seed

    # Set project information
    project_info = config.get("project", {})
    project_name = project_info.get("name", "Plant Disease Detection")
    project_version = project_info.get("version", "1.0.0")

    print(f"Starting {project_name} v{project_version} Training")

    # Set random seed
    seed = config.get("seed", 42)
    set_global_seeds(seed)

    # Configure hardware
    hardware_info = configure_hardware(config)

    # Load data
    data_loader = DataLoader(config)
    train_data, val_data, test_data, class_names = data_loader.load_data(args.data_dir)

    print(f"Loaded dataset with {len(class_names)} classes")

    # Create enhanced model factory
    model_factory = EnhancedModelFactory()

    # Get model architecture
    model_name = args.model
    num_classes = len(class_names)

    print(f"Creating model: {model_name}")

    # Determine whether to use attention or an enhanced model variant
    if args.use_enhanced:
        # Use a pre-configured enhanced model variant (like ResNet50_CBAM)
        # The model_name should be one of the enhanced model names
        model = model_factory.get_model(model_name, num_classes)
    else:
        # Get standard model with optional attention type
        attention_type = args.attention or config.get("training", {}).get(
            "attention_type", None
        )
        model = model_factory.get_model(
            model_name, num_classes, attention_type=attention_type
        )

    # Get training parameters
    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 32)
    epochs = training_config.get("epochs", 50)
    learning_rate = training_config.get("learning_rate", 0.001)
    optimizer_name = training_config.get("optimizer", "adam").lower()

    # Set up optimizer
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        momentum = training_config.get("momentum", 0.9)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=learning_rate, momentum=momentum
        )
    elif optimizer_name == "rmsprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Apply discriminative learning rates if configured
    if config.get("training", {}).get("discriminative_lr", {}).get("enabled", False):
        print("Applying discriminative learning rates")
        discriminative_config = config.get("training", {}).get("discriminative_lr", {})
        base_lr = discriminative_config.get("base_lr", learning_rate)
        factor = discriminative_config.get("factor", 0.3)

        # Use the factory method to get layer-specific learning rates
        layer_lrs = model_factory.get_discriminative_learning_rates(
            model, base_lr=base_lr, factor=factor
        )

        # Log the learning rates for different layers
        print(
            f"Using discriminative learning rates with base_lr={base_lr}, factor={factor}"
        )
        print(
            f"Layer learning rates range from {min([lr for _, lr in layer_lrs])} to {max([lr for _, lr in layer_lrs])}"
        )

    # Run learning rate finder if configured
    if config.get("training", {}).get("lr_finder", {}).get("enabled", True):
        print("Running learning rate finder...")
        lr_config = config.get("training", {}).get("lr_finder", {})

        # Get parameters for LR finder
        min_lr = lr_config.get("min_lr", 1e-7)
        max_lr = lr_config.get("max_lr", 1.0)
        num_steps = lr_config.get("num_steps", 100)

        # Compile model temporarily for LR finder
        model.compile(
            optimizer=optimizer,
            loss=training_config.get("loss", "categorical_crossentropy"),
            metrics=training_config.get("metrics", ["accuracy"]),
        )

        # Create a limited dataset for LR finder
        lr_dataset = train_data.take(num_steps)

        # Run LR finder
        try:
            _, _, optimal_lr = find_optimal_learning_rate(
                model,
                lr_dataset,
                optimizer=optimizer,
                min_lr=min_lr,
                max_lr=max_lr,
                num_steps=num_steps,
                plot_results=True,
            )

            # Update learning rate in optimizer if configured
            if lr_config.get("use_found_lr", True):
                K.set_value(optimizer.lr, optimal_lr)
                print(f"Setting learning rate to optimal value: {optimal_lr:.2e}")
                # Update the config as well
                config["training"]["learning_rate"] = float(optimal_lr)
        except Exception as e:
            print(f"Error running learning rate finder: {e}")
            print("Continuing with original learning rate")

    # Create run directory for this training
    run_dir = paths.get_model_trial_dir(model_name)
    print(f"Training results will be saved to: {run_dir}")

    # Initialize trainer and train the model
    trainer = Trainer(config)

    # Define additional callbacks
    callbacks = []

    # Add progressive freezing callback if configured
    if config.get("training", {}).get("progressive_freezing", {}).get("enabled", False):
        print("Using progressive layer freezing during training")
        freeze_config = config.get("training", {}).get("progressive_freezing", {})
        freeze_layers = freeze_config.get("freeze_layers", 100)
        finetuning_epochs = freeze_config.get("finetuning_epochs", 5)

        # Use the factory method to create the progressive freezing callback
        progressive_callback = model_factory.get_progressive_freezing_callback(
            model,
            num_layers_to_freeze=freeze_layers,
            finetuning_epochs=finetuning_epochs,
        )
        callbacks.append(progressive_callback)

    start_time = time.time()

    try:
        # Train the model
        model, history, metrics = trainer.train(
            model,
            model_name,
            train_data,
            val_data,
            test_data,
            resume=args.resume,
            callbacks=callbacks,
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")

        # Register model in the registry
        try:
            registry = ModelRegistryManager()
            registry.register_model(model, model_name, metrics, history, run_dir)
            print(f"Model registered in registry")
        except Exception as e:
            print(f"Failed to register model in registry: {e}")

        # Print final metrics
        print(f"\nTraining Summary:")
        print(f"  Model: {model_name}")
        print(f"  Training time: {training_time:.2f} seconds")
        print(f"  Final accuracy: {metrics.get('test_accuracy', 0):.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
```

---

### src/training/__init__.py

```python
```

---

### src/training/batch_trainer.py

```python
"""
Batch trainer module for running multiple model training sessions.
This is extracted from main.py to separate batch training logic from the command-line interface.
"""

import time
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm.auto import tqdm
import tensorflow as tf

from src.config.config import get_paths
from src.utils.logger import Logger
from src.utils.report_generator import ReportGenerator
from src.training.model_trainer import train_model
from src.utils.memory_utils import clean_memory, log_memory_usage, memory_monitoring_decorator


class BatchTrainer:
    """Handles batch training of multiple models with logging and reporting."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the batch trainer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.paths = get_paths()
        self.batch_logger = None
        self.models_to_train = []
        self.results = {}
        self.successful_models = 0
        self.failed_models = 0

    def setup_batch_logging(self) -> None:
        """Set up batch logging with a timestamp-based directory."""
        # Create a directory for batch logs
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_log_dir = self.paths.logs_dir / f"batch_{batch_timestamp}"
        batch_log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize batch logger for overall process tracking
        self.batch_logger = Logger(
            "batch_training",
            log_dir=batch_log_dir,
            config=self.config.get("logging", {}),
            logger_type="batch",
        )

        project_info = self.config.get("project", {})
        project_name = project_info.get("name", "Plant Disease Detection")
        project_version = project_info.get("version", "1.0.0")
        
        self.batch_logger.log_info(f"Starting {project_name} v{project_version}")
        self.batch_logger.log_config(self.config)

    def set_models_to_train(self, models: List[str]) -> None:
        """Set the list of models to train in this batch.
        
        Args:
            models: List of model names to train
        """
        self.models_to_train = models
        if self.batch_logger:
            self.batch_logger.log_info(
                f"Will train {len(models)} models: {', '.join(models)}"
            )

    def run_batch_training(
        self, 
        data_loader: Any,
        model_factory: Any,
        train_data: tf.data.Dataset,
        val_data: tf.data.Dataset,
        test_data: Optional[tf.data.Dataset],
        class_names: Dict[int, str],
        resume: bool = False,
        attention_type: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Run batch training for all specified models.
        
        Args:
            data_loader: DataLoader instance
            model_factory: ModelFactory instance 
            train_data: Training dataset
            val_data: Validation dataset
            test_data: Test dataset (optional)
            class_names: Dictionary mapping class indices to names
            resume: Whether to resume training from latest checkpoint
            attention_type: Type of attention mechanism to use (optional)
            
        Returns:
            Dictionary of results for each model
        """
        start_time = time.time()
        self.results = {}
        self.successful_models = 0
        self.failed_models = 0

        # Train all specified models
        for model_name in (model_pbar := tqdm(self.models_to_train, desc="Models", position=0)):
            model_pbar.set_description(f"Training {model_name}")

            model_start_time = time.time()
            
            # Clean memory before training each model
            clean_memory(clean_gpu=True)
            
            # Log memory usage before training
            self.batch_logger.log_info(f"Starting training for model: {model_name}")
            log_memory_usage(prefix=f"Before training {model_name}: ")
            
            success, metrics = train_model(
                model_name,
                self.config,
                data_loader,
                model_factory,
                train_data,
                val_data,
                test_data,
                class_names,
                self.batch_logger,
                resume=resume,
                attention_type=attention_type,
            )
            
            # Log memory usage after training
            log_memory_usage(prefix=f"After training {model_name}: ")
            
            model_time = time.time() - model_start_time

            self.results[model_name] = metrics
            if success:
                self.successful_models += 1
            else:
                self.failed_models += 1

            status_str = f"{'✓' if success else '✗'} in {model_time:.1f}s"
            model_pbar.set_postfix_str(status_str)

            # Log model completion
            accuracy = (
                metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
                if "error" not in metrics
                else 0
            )
            self.batch_logger.log_info(
                f"Model {model_name} completed - Status: {'Success' if success else 'Failed'}, "
                f"Time: {model_time:.2f}s, Accuracy: {accuracy:.4f}"
            )

        return self.results

    def generate_comparison_report(self) -> Optional[str]:
        """Generate a comparison report for all successfully trained models.
        
        Returns:
            Path to the generated report, or None if no report was generated
        """
        # Don't generate a report if there's only one model or no successful models
        if len(self.results) <= 1 or self.successful_models == 0:
            return None
            
        if not self.config.get("reporting", {}).get("generate_html_report", True):
            return None
            
        try:
            comparison_data = []
            for model_name, metrics in self.results.items():
                if "error" not in metrics:
                    comparison_data.append({"name": model_name, "metrics": metrics})

            if comparison_data:
                report_generator = ReportGenerator(self.config)
                comparison_path = report_generator.generate_comparison_report(
                    comparison_data
                )
                print(f"Model comparison report generated at {comparison_path}")
                if self.batch_logger:
                    self.batch_logger.log_info(
                        f"Model comparison report generated at {comparison_path}"
                    )
                return comparison_path
                
        except Exception as e:
            error_msg = f"Error generating comparison report: {e}"
            print(error_msg)
            if self.batch_logger:
                self.batch_logger.log_error(error_msg)
                
        return None

    def save_batch_summary(self, total_time: float) -> None:
        """Save batch training summary metrics.
        
        Args:
            total_time: Total time spent on batch training in seconds
        """
        batch_metrics = {
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "successful_models": self.successful_models,
            "failed_models": self.failed_models,
            "total_models": len(self.models_to_train),
            "seed": self.config.get("seed", 42),
        }

        # Create detailed model results for batch logging
        for model_name, metrics in self.results.items():
            if "error" in metrics:
                print(f"{model_name}: Failed - {metrics['error']}")
                self.batch_logger.log_info(f"{model_name}: Failed - {metrics['error']}")
                batch_metrics[f"{model_name}_status"] = "failed"
                batch_metrics[f"{model_name}_error"] = metrics["error"]
            else:
                accuracy = metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
                train_time = metrics.get("training_time_seconds", 0)
                print(
                    f"{model_name}: Success - Accuracy: {accuracy:.4f} - Time: {train_time:.2f}s"
                )
                self.batch_logger.log_info(
                    f"{model_name}: Success - Accuracy: {accuracy:.4f} - Time: {train_time:.2f}s"
                )
                batch_metrics[f"{model_name}_status"] = "success"
                batch_metrics[f"{model_name}_accuracy"] = accuracy
                batch_metrics[f"{model_name}_training_time"] = train_time

        # Save final batch metrics
        self.batch_logger.save_final_metrics(batch_metrics)
        
    def cleanup_resources(self) -> None:
        """
        Clean up resources after batch training to prevent memory leaks.
        This should be called after completing or interrupting batch training.
        """
        # Log memory stats before cleanup
        self.batch_logger.log_info("Starting resource cleanup...")
        before_cleanup = log_memory_usage(prefix="Before cleanup: ")
        
        # Clean TensorFlow session and force garbage collection
        clean_memory(clean_gpu=True)
        
        # Release large objects
        if hasattr(self, 'results') and self.results:
            # Keep basic info but release any large data
            for model_name in self.results:
                if 'model' in self.results[model_name]:
                    del self.results[model_name]['model']
                if 'history' in self.results[model_name]:
                    # Retain just the final epoch values
                    history = self.results[model_name]['history']
                    self.results[model_name]['history'] = {k: [v[-1]] for k, v in history.items() if isinstance(v, list)}
        
        # Run another garbage collection pass
        gc.collect()
        
        # Log memory stats after cleanup
        after_cleanup = log_memory_usage(prefix="After cleanup: ")
        
        # Calculate memory freed
        memory_freed = before_cleanup["rss_mb"] - after_cleanup["rss_mb"]
        self.batch_logger.log_info(f"Cleanup complete. Memory freed: {memory_freed:.1f}MB")```

---

### src/training/learning_rate_scheduler.py

```python
"""
Learning rate scheduling utilities for training with advanced schedules.
"""

import tensorflow as tf
import numpy as np
import math
from typing import Dict, Any, Optional, Union, Callable, List


class WarmupScheduler(tf.keras.callbacks.Callback):
    """Learning rate scheduler that implements warmup followed by different decay strategies."""

    def __init__(
        self,
        base_lr: float = 0.001,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        strategy: str = "cosine_decay",
        min_lr: float = 1e-6,
        verbose: int = 1,
    ):
        """Initialize the learning rate scheduler with warmup.

        Args:
            base_lr: Base learning rate after warmup
            warmup_epochs: Number of epochs for warmup phase
            total_epochs: Total number of training epochs
            strategy: Decay strategy after warmup ('cosine_decay', 'exponential_decay',
                      'step_decay', or 'constant')
            min_lr: Minimum learning rate
            verbose: Verbosity level
        """
        super(WarmupScheduler, self).__init__()
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.min_lr = min_lr
        self.verbose = verbose
        self.iterations = 0
        self.history = {}
        self.epochs = 0

    def on_epoch_begin(self, epoch, logs=None):
        """Set the learning rate for the current epoch.

        Args:
            epoch: Current epoch number
            logs: Training logs
        """
        self.epochs = epoch

        if epoch < self.warmup_epochs:
            # Warmup phase: linear increase from 0 to base_lr
            lr = (epoch + 1) / self.warmup_epochs * self.base_lr
        else:
            # Post-warmup phase: use selected decay strategy
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )

            # Ensure progress is in [0, 1]
            progress = min(max(0.0, progress), 1.0)

            if self.strategy == "cosine_decay":
                # Cosine decay from base_lr to min_lr
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                    1 + np.cos(np.pi * progress)
                )
            elif self.strategy == "exponential_decay":
                # Exponential decay from base_lr to min_lr
                decay_rate = np.log(self.min_lr / self.base_lr)
                lr = self.base_lr * np.exp(decay_rate * progress)
            elif self.strategy == "step_decay":
                # Step decay (reduce by 1/10 at 50% and 75% of training)
                if progress >= 0.75:
                    lr = self.base_lr * 0.01
                elif progress >= 0.5:
                    lr = self.base_lr * 0.1
                else:
                    lr = self.base_lr
            else:  # "constant"
                # Constant learning rate after warmup
                lr = self.base_lr

        # Ensure learning rate is at least min_lr
        lr = max(self.min_lr, lr)

        # Set the learning rate - handle LossScaleOptimizer for mixed precision
        try:
            if isinstance(
                self.model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer
            ):
                # For LossScaleOptimizer in mixed precision
                tf.keras.backend.set_value(
                    self.model.optimizer.inner_optimizer.learning_rate, lr
                )
            else:
                # For regular optimizers
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        except Exception as e:
            print(f"Warning: Failed to set learning rate: {e}")
            print(f"Optimizer type: {type(self.model.optimizer).__name__}")

        # Log the learning rate
        self.history.setdefault("lr", []).append(lr)

        if self.verbose > 0:
            print(f"\nEpoch {epoch+1}: LR = {lr:.2e}")

    def on_batch_end(self, batch, logs=None):
        """Update iterations count.

        Args:
            batch: Current batch number
            logs: Training logs
        """
        self.iterations += 1


class OneCycleLRScheduler(tf.keras.callbacks.Callback):
    """One-Cycle Learning Rate Policy.

    This implements the one-cycle learning rate policy from the paper
    "Super-Convergence: Very Fast Training of Neural Networks Using
    Large Learning Rates" by Leslie N. Smith.
    """

    def __init__(
        self,
        max_lr: float,
        steps_per_epoch: int,
        epochs: int,
        min_lr: float = 1e-6,
        div_factor: float = 25.0,
        pct_start: float = 0.3,
        anneal_strategy: str = "cos",
        verbose: int = 1,
    ):
        """Initialize the one-cycle learning rate scheduler.

        Args:
            max_lr: Maximum learning rate
            steps_per_epoch: Number of steps (batches) per epoch
            epochs: Total number of training epochs
            min_lr: Minimum learning rate
            div_factor: Determines the initial learning rate as max_lr / div_factor
            pct_start: Percentage of the cycle where the learning rate increases
            anneal_strategy: Strategy for annealing ('cos' or 'linear')
            verbose: Verbosity level
        """
        super(OneCycleLRScheduler, self).__init__()
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.min_lr = min_lr
        self.div_factor = div_factor
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.verbose = verbose

        # Calculate initial learning rate
        self.initial_lr = max_lr / div_factor

        # Calculate total steps
        self.total_steps = steps_per_epoch * epochs

        # Calculate steps for increasing and decreasing phases
        self.step_size_up = int(self.total_steps * pct_start)
        self.step_size_down = self.total_steps - self.step_size_up

        # Current step counter
        self.step_count = 0

        # Learning rate history
        self.history = {}

    def on_train_begin(self, logs=None):
        """Set initial learning rate at the start of training."""
        tf.keras.backend.set_value(self.model.optimizer.lr, self.initial_lr)

    def on_batch_end(self, batch, logs=None):
        """Update learning rate at the end of each batch.

        Args:
            batch: Current batch number
            logs: Training logs
        """
        # Calculate current learning rate
        if self.step_count <= self.step_size_up:
            # Increasing phase
            progress = self.step_count / self.step_size_up

            if self.anneal_strategy == "cos":
                lr = (
                    self.initial_lr
                    + (self.max_lr - self.initial_lr)
                    * (1 - np.cos(np.pi * progress))
                    / 2
                )
            else:  # linear
                lr = self.initial_lr + (self.max_lr - self.initial_lr) * progress
        else:
            # Decreasing phase
            progress = (self.step_count - self.step_size_up) / self.step_size_down

            if self.anneal_strategy == "cos":
                lr = (
                    self.max_lr
                    - (self.max_lr - self.min_lr) * (1 - np.cos(np.pi * progress)) / 2
                )
            else:  # linear
                lr = self.max_lr - (self.max_lr - self.min_lr) * progress

        # Set learning rate - handle LossScaleOptimizer for mixed precision
        try:
            if isinstance(
                self.model.optimizer, tf.keras.mixed_precision.LossScaleOptimizer
            ):
                # For LossScaleOptimizer in mixed precision
                tf.keras.backend.set_value(
                    self.model.optimizer.inner_optimizer.learning_rate, lr
                )
            else:
                # For regular optimizers
                tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        except Exception as e:
            print(f"Warning: Failed to set learning rate: {e}")
            print(f"Optimizer type: {type(self.model.optimizer).__name__}")

        # Log the learning rate
        self.history.setdefault("lr", []).append(lr)

        # Log every verbose steps
        if self.verbose > 0 and self.step_count % self.verbose == 0:
            print(f"Step {self.step_count}: LR = {lr:.2e}")

        # Increment step counter
        self.step_count += 1


def get_warmup_scheduler(
    config: Dict[str, Any]
) -> Optional[tf.keras.callbacks.Callback]:
    """Create a learning rate scheduler based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Learning rate scheduler callback or None if not configured
    """
    # Extract learning rate configuration
    training_config = config.get("training", {})
    lr_schedule = training_config.get("lr_schedule", {})

    # Check if learning rate scheduling is enabled
    if not lr_schedule.get("enabled", False):
        return None

    # Get schedule type
    schedule_type = lr_schedule.get("type", "warmup_cosine")

    # Get base learning rate
    base_lr = training_config.get("learning_rate", 0.001)

    # Get other parameters
    total_epochs = training_config.get("epochs", 100)
    warmup_epochs = lr_schedule.get("warmup_epochs", 5)
    min_lr = lr_schedule.get("min_lr", 1e-6)

    # Create scheduler based on type
    if schedule_type == "warmup_cosine":
        return WarmupScheduler(
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            strategy="cosine_decay",
            min_lr=min_lr,
        )
    elif schedule_type == "warmup_exponential":
        return WarmupScheduler(
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            strategy="exponential_decay",
            min_lr=min_lr,
        )
    elif schedule_type == "warmup_step":
        return WarmupScheduler(
            base_lr=base_lr,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            strategy="step_decay",
            min_lr=min_lr,
        )
    elif schedule_type == "one_cycle":
        # Calculate steps per epoch (estimate if not provided)
        steps_per_epoch = lr_schedule.get("steps_per_epoch", 100)
        max_lr = lr_schedule.get("max_lr", base_lr * 10)
        div_factor = lr_schedule.get("div_factor", 25.0)
        pct_start = lr_schedule.get("pct_start", 0.3)

        return OneCycleLRScheduler(
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=total_epochs,
            min_lr=min_lr,
            div_factor=div_factor,
            pct_start=pct_start,
        )
    else:
        # Unknown schedule type
        print(f"Unknown learning rate schedule type: {schedule_type}")
        return None
```

---

### src/training/lr_finder.py

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
from scipy.signal import savgol_filter
import time
import logging

logger = logging.getLogger(__name__)


def find_optimal_learning_rate(
    model,
    train_dataset,
    loss_fn=None,
    optimizer=None,
    min_lr=1e-7,
    max_lr=1.0,
    num_steps=100,
    stop_factor=4.0,
    smoothing=True,
    plot_results=True,
):
    """Find optimal learning rate using exponential increase and loss tracking

    Args:
        model: The Keras model to train
        train_dataset: tf.data.Dataset containing training data
        loss_fn: Loss function to use (if None, uses model's compiled loss)
        optimizer: Optimizer to use (if None, uses model's compiled optimizer)
        min_lr: Minimum learning rate to test
        max_lr: Maximum learning rate to test
        num_steps: Number of learning rate steps to test
        stop_factor: Stop if loss exceeds best loss by this factor
        smoothing: Whether to apply smoothing to the loss curve
        plot_results: Whether to generate and display a plot

    Returns:
        Tuple of (learning_rates, losses, optimal_lr)
    """
    # Ensure dataset is batched and has at least num_steps batches
    if hasattr(train_dataset, "cardinality"):
        dataset_size = train_dataset.cardinality().numpy()
        if dataset_size == tf.data.INFINITE_CARDINALITY:
            # Dataset is repeat()-ed, we're good
            pass
        elif dataset_size < num_steps:
            logger.warning(
                f"Dataset only has {dataset_size} batches, but {num_steps} steps requested. "
                f"Creating a repeat()-ed dataset."
            )
            train_dataset = train_dataset.repeat()

    # Get loss function and optimizer from model if not provided
    if loss_fn is None:
        if not hasattr(model, "loss") or model.loss is None:
            raise ValueError(
                "No loss function provided and model is not compiled with a loss function"
            )
        loss_fn = model.loss

    if optimizer is None:
        if not hasattr(model, "optimizer") or model.optimizer is None:
            raise ValueError(
                "No optimizer provided and model is not compiled with an optimizer"
            )
        optimizer = model.optimizer

    logger.info(
        f"Starting learning rate finder from {min_lr:.1e} to {max_lr:.1e} over {num_steps} steps"
    )

    # Create a copy of model weights
    original_weights = model.get_weights()

    # Save original learning rate
    original_lr = K.get_value(optimizer.lr)

    try:
        # Exponential increase factor
        mult_factor = (max_lr / min_lr) ** (1.0 / num_steps)

        # Lists to store learning rates and losses
        learning_rates = []
        losses = []

        # Set initial learning rate
        K.set_value(optimizer.lr, min_lr)

        # Best loss tracking
        best_loss = float("inf")

        # Time tracking
        start_time = time.time()

        # Process batches
        for batch_idx, batch_data in enumerate(train_dataset):
            if batch_idx >= num_steps:
                break

            # Unpack batch data (handle different dataset formats)
            if isinstance(batch_data, tuple) and len(batch_data) == 2:
                inputs, targets = batch_data
            else:
                inputs = batch_data
                targets = None  # For models that don't need separate targets

            # Record current learning rate
            current_lr = K.get_value(optimizer.lr)
            learning_rates.append(current_lr)

            # Train on batch
            with tf.GradientTape() as tape:
                outputs = model(inputs, training=True)

                # Handle different loss function signatures
                if targets is not None:
                    loss = loss_fn(targets, outputs)
                else:
                    loss = loss_fn(outputs)

            # Update weights
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Convert loss to scalar if it's a tensor
            try:
                loss_value = loss.numpy()
            except:
                loss_value = float(loss)

            # Record loss
            losses.append(loss_value)

            # Check for exploding loss
            if loss_value < best_loss:
                best_loss = loss_value

            # Print progress periodically
            if batch_idx % max(1, num_steps // 10) == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Step {batch_idx}/{num_steps}, lr={current_lr:.2e}, "
                    f"loss={loss_value:.4f}, elapsed={elapsed:.1f}s"
                )

            # Stop if loss is exploding
            if loss_value > stop_factor * best_loss or np.isnan(loss_value):
                logger.info(
                    f"Loss is exploding or NaN detected (loss={loss_value:.4f}, best_loss={best_loss:.4f}). "
                    f"Stopping early at step {batch_idx}/{num_steps}."
                )
                break

            # Increase learning rate
            K.set_value(optimizer.lr, current_lr * mult_factor)

        # Restore original weights and learning rate
        model.set_weights(original_weights)
        K.set_value(optimizer.lr, original_lr)

        # Convert to numpy arrays for easier manipulation
        learning_rates = np.array(learning_rates)
        losses = np.array(losses)

        # Filter out NaN and inf values
        valid_indices = ~(np.isnan(losses) | np.isinf(losses))
        learning_rates = learning_rates[valid_indices]
        losses = losses[valid_indices]

        if len(losses) == 0:
            raise ValueError(
                "No valid loss values found. All losses were NaN or infinite."
            )

        # Apply smoothing if requested
        if smoothing and len(losses) > 7:
            try:
                smoothed_losses = savgol_filter(
                    losses, min(7, len(losses) // 2 * 2 - 1), 3
                )
            except Exception as e:
                logger.warning(
                    f"Error applying smoothing filter: {e}. Using raw loss values."
                )
                smoothed_losses = losses
        else:
            smoothed_losses = losses

        # Find the point of steepest descent (minimum gradient)
        try:
            # Calculate the gradients of the loss curve
            gradients = np.gradient(smoothed_losses)

            # Find where the gradient is steepest (most negative)
            optimal_idx = np.argmin(gradients)

            # The optimal lr is typically a bit lower than the minimum gradient point
            optimal_lr = (
                learning_rates[optimal_idx] / 10.0
            )  # Division by 10 is a rule of thumb
        except Exception as e:
            logger.warning(
                f"Error finding optimal learning rate: {e}. Using fallback method."
            )
            # Fallback: Find point with fastest loss decrease
            loss_ratios = losses[1:] / losses[:-1]
            fastest_decrease_idx = np.argmin(loss_ratios)
            if fastest_decrease_idx < len(learning_rates) - 1:
                optimal_lr = learning_rates[fastest_decrease_idx]
            else:
                optimal_lr = learning_rates[len(learning_rates) // 2] / 10.0

        # Ensure we found a reasonable learning rate
        if optimal_lr <= min_lr or optimal_lr >= max_lr:
            logger.warning(
                f"Optimal learning rate ({optimal_lr:.2e}) is at or outside the bounds "
                f"of the tested range ({min_lr:.2e} - {max_lr:.2e}). Consider adjusting the range."
            )

        # Plot the results if requested
        if plot_results:
            plot_lr_finder_results(learning_rates, losses, smoothed_losses, optimal_lr)

        logger.info(
            f"Learning rate finder complete. Optimal learning rate: {optimal_lr:.2e}"
        )

        return learning_rates, losses, optimal_lr

    except Exception as e:
        # Restore original weights and learning rate in case of error
        model.set_weights(original_weights)
        K.set_value(optimizer.lr, original_lr)
        logger.error(f"Error during learning rate finding: {e}")
        raise


def plot_lr_finder_results(
    learning_rates, losses, smoothed_losses=None, optimal_lr=None
):
    """
    Plot the results of the learning rate finder.

    Args:
        learning_rates: List of learning rates
        losses: List of loss values
        smoothed_losses: List of smoothed loss values (optional)
        optimal_lr: The determined optimal learning rate (optional)
    """
    plt.figure(figsize=(12, 6))

    # Plot raw losses
    plt.plot(learning_rates, losses, "b-", alpha=0.3, label="Raw loss")

    # Plot smoothed losses if available
    if smoothed_losses is not None:
        plt.plot(learning_rates, smoothed_losses, "r-", label="Smoothed loss")

    # Mark the optimal learning rate if provided
    if optimal_lr is not None:
        plt.axvline(
            x=optimal_lr,
            color="green",
            linestyle="--",
            label=f"Optimal LR: {optimal_lr:.2e}",
        )

    plt.xscale("log")
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # Save the figure
    try:
        plt.savefig("learning_rate_finder.png", dpi=300, bbox_inches="tight")
        logger.info("Learning rate finder plot saved as 'learning_rate_finder.png'")
    except Exception as e:
        logger.warning(f"Could not save learning rate finder plot: {e}")

    plt.tight_layout()
    plt.show()


def find_and_set_learning_rate(model, train_dataset, optimizer=None, **kwargs):
    """
    Find the optimal learning rate and set it in the model's optimizer.

    Args:
        model: The Keras model
        train_dataset: Training dataset
        optimizer: Optional optimizer (uses model's optimizer if None)
        **kwargs: Additional arguments to pass to find_optimal_learning_rate

    Returns:
        The optimal learning rate
    """
    # Get the optimizer from model if not provided
    if optimizer is None:
        if not hasattr(model, "optimizer") or model.optimizer is None:
            raise ValueError("Model is not compiled with an optimizer")
        optimizer = model.optimizer

    # Run the learning rate finder
    logger.info("Running learning rate finder...")
    _, _, optimal_lr = find_optimal_learning_rate(
        model, train_dataset, optimizer=optimizer, **kwargs
    )

    # Set the found learning rate
    logger.info(f"Setting optimizer learning rate to {optimal_lr:.2e}")
    K.set_value(optimizer.lr, optimal_lr)

    return optimal_lr


class LearningRateFinderCallback(tf.keras.callbacks.Callback):
    """
    Callback to find optimal learning rate before training starts.

    This callback runs the learning rate finder for one epoch before the actual training begins,
    then sets the optimal learning rate for the optimizer.
    """

    def __init__(
        self,
        min_lr=1e-7,
        max_lr=1.0,
        num_steps=100,
        stop_factor=4.0,
        use_validation=False,
        plot_results=True,
        set_lr=True,
    ):
        """
        Initialize the learning rate finder callback.

        Args:
            min_lr: Minimum learning rate to test
            max_lr: Maximum learning rate to test
            num_steps: Number of steps for LR range test
            stop_factor: Stop if loss exceeds best loss by this factor
            use_validation: Whether to use validation data if available
            plot_results: Whether to plot the results
            set_lr: Whether to automatically set the learning rate
        """
        super(LearningRateFinderCallback, self).__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.num_steps = num_steps
        self.stop_factor = stop_factor
        self.use_validation = use_validation
        self.plot_results = plot_results
        self.set_lr = set_lr
        self.optimal_lr = None
        self.learning_rates = None
        self.losses = None

    def on_train_begin(self, logs=None):
        """Run the learning rate finder before training starts."""
        logger.info(
            "LearningRateFinderCallback: Finding optimal learning rate before training..."
        )

        # Save original learning rate
        self.original_lr = K.get_value(self.model.optimizer.lr)

        # Use validation data if available and requested
        if (
            self.use_validation
            and hasattr(self.model, "validation_data")
            and self.model.validation_data is not None
        ):
            dataset = self.model.validation_data
            logger.info("Using validation data for learning rate finder")
        else:
            dataset = self.params["train_data"]
            logger.info("Using training data for learning rate finder")

        # Run the learning rate finder
        self.learning_rates, self.losses, self.optimal_lr = find_optimal_learning_rate(
            self.model,
            dataset,
            min_lr=self.min_lr,
            max_lr=self.max_lr,
            num_steps=self.num_steps,
            stop_factor=self.stop_factor,
            plot_results=self.plot_results,
        )

        # Set the learning rate if requested
        if self.set_lr:
            logger.info(
                f"Setting learning rate to optimal value: {self.optimal_lr:.2e}"
            )
            K.set_value(self.model.optimizer.lr, self.optimal_lr)
        else:
            # Restore original learning rate
            logger.info(f"Restoring original learning rate: {self.original_lr:.2e}")
            K.set_value(self.model.optimizer.lr, self.original_lr)

        # Log the finding
        logs = logs or {}
        logs["optimal_lr"] = self.optimal_lr

        return logs


def find_batch_aware_lr(
    model,
    train_dataset,
    batch_size_range=(16, 256),
    lr_range=(1e-6, 1e-1),
    n_batch_sizes=5,
    plot_results=True,
):
    """
    Find the optimal learning rate for different batch sizes.

    This function explores the relationship between batch size and optimal learning rate,
    which often follows a linear relationship.

    Args:
        model: The Keras model
        train_dataset: Training dataset (unbatched)
        batch_size_range: Tuple of (min_batch_size, max_batch_size)
        lr_range: Tuple of (min_lr, max_lr) for the search
        n_batch_sizes: Number of batch sizes to test
        plot_results: Whether to plot the results

    Returns:
        Dictionary mapping batch sizes to their optimal learning rates
    """
    min_batch, max_batch = batch_size_range

    # Generate batch sizes in log space
    batch_sizes = np.unique(
        np.logspace(np.log10(min_batch), np.log10(max_batch), n_batch_sizes).astype(int)
    )

    # Make sure train_dataset is unbatched
    if hasattr(train_dataset, "unbatch"):
        unbatched_dataset = train_dataset.unbatch()
    else:
        unbatched_dataset = train_dataset

    results = {}

    for batch_size in batch_sizes:
        logger.info(f"Finding optimal learning rate for batch size {batch_size}...")

        # Create a batched dataset with this batch size
        batched_dataset = unbatched_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        # Find the optimal learning rate
        _, _, optimal_lr = find_optimal_learning_rate(
            model,
            batched_dataset,
            min_lr=lr_range[0],
            max_lr=lr_range[1],
            plot_results=False,
        )

        results[int(batch_size)] = optimal_lr
        logger.info(f"Batch size {batch_size}: optimal LR = {optimal_lr:.2e}")

    if plot_results:
        plt.figure(figsize=(10, 6))
        batch_sizes_arr = np.array(list(results.keys()))
        lr_arr = np.array(list(results.values()))

        plt.plot(batch_sizes_arr, lr_arr, "o-", markersize=10)
        plt.xscale("log", base=2)
        plt.yscale("log", base=10)
        plt.xlabel("Batch Size (log scale)")
        plt.ylabel("Optimal Learning Rate (log scale)")
        plt.title("Batch Size vs. Optimal Learning Rate")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)

        # Linear fit in log-log space
        if len(batch_sizes_arr) >= 2:
            # Linear fit in log-log space - relationship is often LR ∝ batch_size
            coeffs = np.polyfit(np.log2(batch_sizes_arr), np.log10(lr_arr), 1)
            slope = coeffs[0]

            # Plot the fit
            x_fit = np.linspace(min(batch_sizes_arr), max(batch_sizes_arr), 100)
            y_fit = 10 ** (coeffs[0] * np.log2(x_fit) + coeffs[1])
            plt.plot(x_fit, y_fit, "r--", label=f"Fit: LR ∝ batch_size^{slope:.2f}")
            plt.legend()

        try:
            plt.savefig("batch_aware_lr.png", dpi=300, bbox_inches="tight")
            logger.info("Batch-aware learning rate plot saved as 'batch_aware_lr.png'")
        except Exception as e:
            logger.warning(f"Could not save batch-aware learning rate plot: {e}")

        plt.show()

    return results


def calculate_learning_rate(
    batch_size, results=None, reference_batch=32, reference_lr=1e-3
):
    """
    Calculate the appropriate learning rate for a given batch size based on the linear scaling rule.

    If results from find_batch_aware_lr are provided, uses interpolation from those results.
    Otherwise, uses the linear scaling rule: LR ∝ batch_size.

    Args:
        batch_size: The batch size to calculate learning rate for
        results: Optional dictionary of {batch_size: optimal_lr} from find_batch_aware_lr
        reference_batch: Reference batch size for linear scaling rule
        reference_lr: Reference learning rate for linear scaling rule

    Returns:
        The calculated learning rate for the given batch size
    """
    if results is not None and len(results) >= 2:
        # Use interpolation if we have enough data points
        batch_sizes = np.array(list(results.keys()))
        learning_rates = np.array(list(results.values()))

        # Use log-log interpolation
        log_batch_sizes = np.log2(batch_sizes)
        log_learning_rates = np.log10(learning_rates)

        # Find the interpolation coefficient (slope in log-log space)
        coeffs = np.polyfit(log_batch_sizes, log_learning_rates, 1)

        # Calculate the interpolated learning rate
        log_lr = coeffs[0] * np.log2(batch_size) + coeffs[1]
        return 10**log_lr
    else:
        # Use simple linear scaling rule: LR ∝ batch_size
        return reference_lr * (batch_size / reference_batch)


# Utility to create a function-based learning rate schedule using the finder results
def create_lr_schedule_from_finder(
    min_lr, max_lr, steps_per_epoch, epochs, warmup_epochs=5, decay_epochs=None
):
    """
    Create a learning rate schedule function based on learning rate finder results.

    Uses warmup followed by cosine decay:
    - Linear warmup from min_lr to max_lr over warmup_epochs
    - Cosine decay from max_lr to min_lr over remaining epochs

    Args:
        min_lr: Minimum learning rate (usually from learning rate finder)
        max_lr: Maximum learning rate (usually from learning rate finder)
        steps_per_epoch: Number of steps per epoch
        epochs: Total number of epochs
        warmup_epochs: Number of epochs for linear warmup
        decay_epochs: Number of epochs for decay (defaults to epochs - warmup_epochs)

    Returns:
        A function that takes (epoch, lr) and returns the new learning rate
    """
    if decay_epochs is None:
        decay_epochs = epochs - warmup_epochs

    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_schedule(epoch, lr):
        # Convert epoch to step for more granular control
        step = epoch * steps_per_epoch

        # Linear warmup phase
        if step < warmup_steps:
            return min_lr + (max_lr - min_lr) * (step / warmup_steps)

        # Cosine decay phase
        decay_steps = total_steps - warmup_steps
        decay_step = step - warmup_steps

        cosine_decay = 0.5 * (1 + np.cos(np.pi * decay_step / decay_steps))
        return min_lr + (max_lr - min_lr) * cosine_decay

    return lr_schedule


def onecycle_lr_schedule(initial_lr, max_lr, total_steps, pct_start=0.3):
    """
    Create a One-Cycle learning rate scheduler function.

    Args:
        initial_lr: Starting/ending learning rate
        max_lr: Maximum learning rate in the middle of the cycle
        total_steps: Total number of training steps
        pct_start: Percentage of cycle spent increasing LR

    Returns:
        A function for use with LearningRateScheduler callback
    """

    def schedule(step):
        # Calculate the current position in the cycle
        if step < pct_start * total_steps:
            # Increasing phase
            pct_progress = step / (pct_start * total_steps)
            return initial_lr + (max_lr - initial_lr) * pct_progress
        else:
            # Decreasing phase
            pct_progress = (step - pct_start * total_steps) / (
                (1 - pct_start) * total_steps
            )
            return max_lr - (max_lr - initial_lr) * pct_progress

    return tf.keras.callbacks.LearningRateScheduler(schedule)


class CyclicalLearningRateCallback(tf.keras.callbacks.Callback):
    """
    A callback to implement Cyclical Learning Rate policies during training.

    Supports "triangular", "triangular2", and "exp_range" policies.
    """

    def __init__(self, base_lr, max_lr, step_size, mode="triangular", gamma=0.99994):
        """
        Initialize the cyclical learning rate callback.

        Args:
            base_lr: Minimum learning rate
            max_lr: Maximum learning rate
            step_size: Number of training iterations per half cycle
            mode: One of {"triangular", "triangular2", "exp_range"}
            gamma: Constant for "exp_range" mode, controls decay rate
        """
        super(CyclicalLearningRateCallback, self).__init__()
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        self.iteration = 0
        self.history = {}

    def clr(self):
        """Calculate the current learning rate"""
        cycle = np.floor(1 + self.iteration / (2 * self.step_size))
        x = np.abs(self.iteration / self.step_size - 2 * cycle + 1)

        if self.mode == "triangular":
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x))
        elif self.mode == "triangular2":
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                0, (1 - x)
            ) / (2 ** (cycle - 1))
        elif self.mode == "exp_range":
            lr = self.base_lr + (self.max_lr - self.base_lr) * np.maximum(
                0, (1 - x)
            ) * (self.gamma**self.iteration)
        else:
            raise ValueError(
                f"Mode {self.mode} not supported. Use one of: triangular, triangular2, exp_range"
            )

        return lr

    def on_train_begin(self, logs=None):
        """Initialize at the start of training"""
        # Start from baseline LR
        K.set_value(self.model.optimizer.lr, self.base_lr)

    def on_batch_end(self, batch, logs=None):
        """Update learning rate after each batch"""
        self.iteration += 1
        lr = self.clr()
        K.set_value(self.model.optimizer.lr, lr)

        # Store in history
        self.history.setdefault("lr", []).append(lr)
        if logs:
            self.history.setdefault("loss", []).append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs=None):
        """Log learning rate at the end of each epoch"""
        lr = K.get_value(self.model.optimizer.lr)
        logger.info(f"Epoch {epoch+1}: Cyclical learning rate = {lr:.2e}")


class AdaptiveLearningRateCallback(tf.keras.callbacks.Callback):
    """
    A callback that adapts the learning rate based on training dynamics.

    This callback monitors training metrics and adjusts the learning rate
    accordingly, reducing it when progress stalls or increasing it slightly
    when progress is consistent.
    """

    def __init__(
        self,
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_delta=1e-4,
        min_lr=1e-6,
        max_lr=1.0,
        increase_factor=1.05,
        increase_patience=5,
        cooldown=2,
        verbose=1,
    ):
        """
        Initialize the adaptive learning rate callback.

        Args:
            monitor: Metric to monitor
            factor: Factor by which to reduce learning rate
            patience: Number of epochs with no improvement before reducing LR
            min_delta: Minimum change to qualify as improvement
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate
            increase_factor: Factor by which to increase learning rate
            increase_patience: Number of epochs with consistent improvement before increasing LR
            cooldown: Number of epochs to wait after a LR change
            verbose: Verbosity level
        """
        super(AdaptiveLearningRateCallback, self).__init__()
        self.monitor = monitor
        self.factor = factor
        self.patience = patience
        self.min_delta = min_delta
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.increase_factor = increase_factor
        self.increase_patience = increase_patience
        self.cooldown = cooldown
        self.verbose = verbose

        self.cooldown_counter = 0
        self.wait = 0
        self.increase_wait = 0
        self.best = float("inf") if "loss" in monitor else -float("inf")
        self.monitor_op = np.less if "loss" in monitor else np.greater

    def on_epoch_end(self, epoch, logs=None):
        """Check metrics and adjust learning rate if needed"""
        logs = logs or {}
        current = logs.get(self.monitor)

        if current is None:
            logger.warning(f"AdaptiveLR: {self.monitor} metric not found in logs!")
            return

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0

        # Get current learning rate
        lr = K.get_value(self.model.optimizer.lr)

        # Check if we're better than the previous best
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            self.increase_wait += 1

            # Check if we should increase the learning rate
            if (
                self.increase_wait >= self.increase_patience
                and self.cooldown_counter == 0
            ):
                new_lr = min(lr * self.increase_factor, self.max_lr)

                if new_lr > lr:
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        logger.info(
                            f"Epoch {epoch+1}: AdaptiveLR increasing learning rate to {new_lr:.2e}"
                        )

                    self.cooldown_counter = self.cooldown
                    self.increase_wait = 0
        else:
            self.wait += 1
            self.increase_wait = 0

            # Check if we should decrease the learning rate
            if self.wait >= self.patience and self.cooldown_counter == 0:
                new_lr = max(lr * self.factor, self.min_lr)

                if new_lr < lr:
                    K.set_value(self.model.optimizer.lr, new_lr)
                    if self.verbose > 0:
                        logger.info(
                            f"Epoch {epoch+1}: AdaptiveLR reducing learning rate to {new_lr:.2e}"
                        )

                    self.cooldown_counter = self.cooldown
                    self.wait = 0
```

---

### src/training/model_trainer.py

```python
"""
Model trainer module for handling individual model training processes.
This is extracted from main.py to separate the training logic from the command-line interface.
"""

import os
import time
import tensorflow as tf
from pathlib import Path
import numpy as np
from typing import Dict, Tuple, List, Any, Optional, Union, Callable

from src.config.config import get_paths
from src.utils.logger import Logger
from src.training.lr_finder import find_optimal_learning_rate
from src.utils.report_generator import ReportGenerator
from src.model_registry.registry_manager import ModelRegistryManager


def train_model(
    model_name: str,
    config: Dict[str, Any],
    data_loader: Any,
    model_factory: Any,
    train_data: tf.data.Dataset,
    val_data: tf.data.Dataset,
    test_data: Optional[tf.data.Dataset] = None,
    class_names: Optional[Dict[int, str]] = None,
    batch_logger: Optional[Logger] = None,
    resume: bool = False,
    attention_type: Optional[str] = None,
) -> Tuple[bool, Dict[str, Any]]:
    """Train a single model and return results
    
    Args:
        model_name: Name of the model to train
        config: Configuration dictionary
        data_loader: DataLoader instance
        model_factory: ModelFactory instance
        train_data: Training dataset
        val_data: Validation dataset
        test_data: Test dataset (optional)
        class_names: Dictionary mapping class indices to names
        batch_logger: Logger for batch operations (optional)
        resume: Whether to resume training from latest checkpoint
        attention_type: Type of attention mechanism to use (optional)
        
    Returns:
        Tuple of (success_flag, metrics_dict)
        
    Raises:
        ValueError: If model configuration is invalid
        RuntimeError: If an error occurs during model creation or training
    """
    print(f"\n{'='*80}")
    print(f"Training model: {model_name}")
    print(f"{'='*80}")

    if batch_logger:
        batch_logger.log_info(f"Starting training for model: {model_name}")

    try:
        # Get model hyperparameters (combining defaults with model-specific)
        from src.config.config_loader import ConfigLoader
        config_loader = ConfigLoader()
        hyperparams = config_loader.get_hyperparameters(model_name, config)

        # Update config with these hyperparameters
        training_config = config.get("training", {}).copy()
        training_config.update(hyperparams)
        config["training"] = training_config

        print(f"Training configuration for {model_name}:")
        for key, value in training_config.items():
            print(f"  {key}: {value}")

        if batch_logger:
            batch_logger.log_info(
                f"Hyperparameters for {model_name}: {training_config}"
            )

        # Create model using simplified factory
        if attention_type:
            print(f"Using {model_name} with {attention_type} attention")
            if batch_logger:
                batch_logger.log_info(f"Using {model_name} with {attention_type} attention")
            model = model_factory.create_model(
                model_name=model_name, 
                num_classes=len(class_names) if class_names else None,
                attention_type=attention_type
            )
        else:
            # Get model from config (which may include attention if it's in the model name)
            print(f"Creating model {model_name} from configuration")
            if batch_logger:
                batch_logger.log_info(f"Creating model {model_name} from configuration")
            model = model_factory.get_model_from_config(
                model_name, 
                num_classes=len(class_names) if class_names else None
            )

        # Apply learning rate finder if configured
        if config.get("training", {}).get("lr_finder", {}).get("enabled", False):
            print("Running learning rate finder...")
            if batch_logger:
                batch_logger.log_info("Running learning rate finder...")

            lr_config = config.get("training", {}).get("lr_finder", {})
            min_lr = lr_config.get("min_lr", 1e-7)
            max_lr = lr_config.get("max_lr", 1.0)
            num_steps = lr_config.get("num_steps", 100)

            # Initialize optimizer for LR finder
            learning_rate = config.get("training", {}).get("learning_rate", 0.001)
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Compile model temporarily for LR finder
            model.compile(
                optimizer=optimizer,
                loss=config.get("training", {}).get("loss", "categorical_crossentropy"),
                metrics=config.get("training", {}).get("metrics", ["accuracy"]),
            )

            # Create a limited dataset for LR finder
            lr_dataset = train_data.take(num_steps)

            try:
                # Run LR finder
                _, _, optimal_lr = find_optimal_learning_rate(
                    model,
                    lr_dataset,
                    optimizer=optimizer,
                    min_lr=min_lr,
                    max_lr=max_lr,
                    num_steps=num_steps,
                    plot_results=True,
                )

                # Update learning rate in configuration if requested
                if lr_config.get("use_found_lr", True):
                    print(f"Setting learning rate to optimal value: {optimal_lr:.2e}")
                    if batch_logger:
                        batch_logger.log_info(
                            f"Setting learning rate to optimal value: {optimal_lr:.2e}"
                        )
                    config["training"]["learning_rate"] = float(optimal_lr)
            except Exception as e:
                print(
                    f"Error in learning rate finder: {e}, continuing with original learning rate"
                )
                if batch_logger:
                    batch_logger.log_warning(f"Error in learning rate finder: {e}")

        # Create the trainer and train the model
        from src.training.trainer import Trainer
        trainer = Trainer(config)
        model, history, metrics = trainer.train(
            model, model_name, train_data, val_data, test_data, resume=resume
        )

        # Generate report if enabled
        if config.get("reporting", {}).get("generate_html_report", True):
            report_generator = ReportGenerator(config)
            run_dir = metrics.get("run_dir", "")
            report_path = report_generator.generate_model_report(
                model_name, run_dir, metrics, history, class_names
            )
            print(f"Report generated at {report_path}")

            if batch_logger:
                batch_logger.log_info(
                    f"Report for {model_name} generated at {report_path}"
                )

        # Clean up memory
        del model
        tf.keras.backend.clear_session()
        
        # Explicitly run garbage collection
        import gc
        gc.collect()

        print(f"Training completed for {model_name}")
        accuracy = metrics.get("test_accuracy", metrics.get("val_accuracy", 0))
        training_time = metrics.get("training_time_seconds", 0)

        print(f"Final accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} seconds")

        if batch_logger:
            batch_logger.log_info(f"Model {model_name} training successful")
            batch_logger.log_info(f"Final accuracy: {accuracy:.4f}")
            batch_logger.log_info(f"Training time: {training_time:.2f} seconds")

            # Log summary metrics for the batch report
            batch_summary = {
                f"{model_name}_accuracy": accuracy,
                f"{model_name}_training_time": training_time,
            }
            batch_logger.log_metrics(batch_summary)

        return True, metrics

    except Exception as e:
        error_msg = f"Error training model {model_name}: {e}"
        print(error_msg)
        import traceback

        trace = traceback.format_exc()
        print(trace)

        if batch_logger:
            batch_logger.log_error(error_msg)
            batch_logger.log_error(trace)

        # Clean up memory even on failure
        try:
            del model
        except:
            pass
        
        tf.keras.backend.clear_session()
        import gc
        gc.collect()

        return False, {"error": str(e), "traceback": trace}```

---

### src/training/trainer.py

```python
# src/training/trainer.py

import os
import time
import tensorflow as tf
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import numpy as np

from src.utils.logger import Logger
from src.config.config import get_paths
from src.model_registry.registry_manager import ModelRegistryManager
from src.utils.seed_utils import set_global_seeds
from src.training.learning_rate_scheduler import get_warmup_scheduler


class ProgressBarCallback(tf.keras.callbacks.Callback):
    """Custom callback for displaying training progress with tqdm"""

    def __init__(self, epochs, verbose=1):
        super(ProgressBarCallback, self).__init__()
        self.epochs = epochs
        self.verbose = verbose
        self.epoch_pbar = None
        self.batch_pbar = None

    def on_train_begin(self, logs=None):
        if self.verbose:
            self.epoch_pbar = tqdm(total=self.epochs, desc="Epochs", position=0)

    def on_train_end(self, logs=None):
        if self.verbose and self.epoch_pbar is not None:
            self.epoch_pbar.close()
            self.epoch_pbar = None

    def on_epoch_begin(self, epoch, logs=None):
        if self.verbose and hasattr(self.model, "train_step_count"):
            steps = getattr(self.model, "train_step_count")
            if self.batch_pbar is not None:
                self.batch_pbar.close()
            self.batch_pbar = tqdm(
                total=steps,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                position=1,
                leave=False,
            )

    def on_epoch_end(self, epoch, logs=None):
        if self.verbose:
            if self.epoch_pbar is not None:
                self.epoch_pbar.update(1)
                # Print metrics
                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in logs.items()])
                self.epoch_pbar.set_postfix_str(metrics_str)

            if self.batch_pbar is not None:
                self.batch_pbar.close()
                self.batch_pbar = None

    def on_train_batch_end(self, batch, logs=None):
        if self.verbose and self.batch_pbar is not None:
            self.batch_pbar.update(1)
            if logs:
                # Show only loss and accuracy in batch progress
                metrics_to_show = {}
                if "loss" in logs:
                    metrics_to_show["loss"] = logs["loss"]
                if "accuracy" in logs:
                    metrics_to_show["acc"] = logs["accuracy"]

                if metrics_to_show:
                    metrics_str = ", ".join(
                        [f"{k}: {v:.4f}" for k, v in metrics_to_show.items()]
                    )
                    self.batch_pbar.set_postfix_str(metrics_str)


class Trainer:
    def __init__(self, config=None):
        """Initialize the trainer

        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = None
        self.paths = get_paths()

        # Set random seeds if specified
        if "seed" in self.config:
            set_global_seeds(self.config["seed"])

    def _apply_gradient_clipping(self, optimizer, clip_norm=None, clip_value=None):
        """
        Apply gradient clipping to an optimizer for TensorFlow 2.16.2

        Args:
            optimizer: The optimizer to apply clipping to
            clip_norm: Value for gradient norm clipping (global)
            clip_value: Value for gradient value clipping (per-variable)

        Returns:
            Optimizer with gradient clipping applied
        """
        if clip_norm is None and clip_value is None:
            return optimizer

        self.train_logger.log_info(f"TensorFlow version: {tf.__version__}")
        self.train_logger.log_info(f"Optimizer type: {type(optimizer).__name__}")

        # Check if we have a LossScaleOptimizer wrapper (for mixed precision)
        if hasattr(optimizer, "inner_optimizer"):
            inner_optimizer = optimizer.inner_optimizer
            self.train_logger.log_info(
                f"Inner optimizer type: {type(inner_optimizer).__name__}"
            )
        else:
            inner_optimizer = optimizer

        if clip_norm is not None:
            self.train_logger.log_info(
                f"Using gradient norm clipping with value {clip_norm}"
            )

            try:
                # For TF 2.16.2, use the appropriate method based on optimizer type
                if hasattr(inner_optimizer, "clipnorm"):
                    # Direct attribute setting works for many optimizers
                    inner_optimizer.clipnorm = clip_norm
                    self.train_logger.log_info("Applied clipnorm directly to optimizer")
                    return optimizer
                elif hasattr(inner_optimizer, "with_clipnorm"):
                    # Use with_clipnorm method if available
                    if hasattr(optimizer, "inner_optimizer"):
                        # For LossScaleOptimizer, need to set inner_optimizer and recreate
                        new_inner = inner_optimizer.with_clipnorm(clip_norm)
                        # Recreate the LossScaleOptimizer with the new inner optimizer
                        from tensorflow.keras import mixed_precision

                        new_optimizer = mixed_precision.LossScaleOptimizer(new_inner)
                        self.train_logger.log_info(
                            "Applied clipnorm to inner optimizer with with_clipnorm()"
                        )
                        return new_optimizer
                    else:
                        # For regular optimizers
                        new_optimizer = inner_optimizer.with_clipnorm(clip_norm)
                        self.train_logger.log_info(
                            "Applied clipnorm with with_clipnorm()"
                        )
                        return new_optimizer
                else:
                    # Use global norm clipping as a last resort
                    try:

                        orig_apply_gradients = inner_optimizer.apply_gradients

                        def apply_gradients_with_clip(grads_and_vars, **kwargs):
                            grads, vars = zip(*grads_and_vars)
                            grads, _ = tf.clip_by_global_norm(grads, clip_norm)
                            return orig_apply_gradients(zip(grads, vars), **kwargs)

                        inner_optimizer.apply_gradients = apply_gradients_with_clip
                        self.train_logger.log_info(
                            "Applied clipnorm by modifying apply_gradients method"
                        )
                        return optimizer
                    except Exception as e:
                        self.train_logger.log_warning(
                            f"Failed to apply custom gradient clipping: {e}"
                        )

            except Exception as e:
                self.train_logger.log_warning(
                    f"Failed to apply gradient norm clipping: {e}"
                )
                self.train_logger.log_warning(
                    "Training will proceed without gradient clipping."
                )

        if clip_value is not None:
            self.train_logger.log_info(
                f"Using gradient value clipping with value {clip_value}"
            )

            try:
                # For TF 2.16.2 optimizers
                if hasattr(inner_optimizer, "clipvalue"):
                    inner_optimizer.clipvalue = clip_value
                    self.train_logger.log_info(
                        "Applied clipvalue directly to optimizer"
                    )
                    return optimizer
                elif hasattr(optimizer, "with_clipvalue"):
                    optimizer = optimizer.with_clipvalue(clip_value)
                    self.train_logger.log_info(
                        "Applied clipvalue using with_clipvalue() method"
                    )
                else:
                    # Use custom clipping as a last resort
                    try:

                        orig_apply_gradients = inner_optimizer.apply_gradients

                        def apply_gradients_with_clip(grads_and_vars, **kwargs):
                            clipped_grads_and_vars = [
                                (
                                    (tf.clip_by_value(g, -clip_value, clip_value), v)
                                    if g is not None
                                    else (None, v)
                                )
                                for g, v in grads_and_vars
                            ]
                            return orig_apply_gradients(
                                clipped_grads_and_vars, **kwargs
                            )

                        inner_optimizer.apply_gradients = apply_gradients_with_clip
                        self.train_logger.log_info(
                            "Applied clipvalue by modifying apply_gradients method"
                        )
                        return optimizer
                    except Exception as e:
                        self.train_logger.log_warning(
                            f"Failed to apply custom gradient value clipping: {e}"
                        )
            except Exception as e:
                self.train_logger.log_warning(
                    f"Failed to apply gradient value clipping: {e}"
                )
                self.train_logger.log_warning(
                    "Training will proceed without value clipping."
                )

        return optimizer

    def train(
        self,
        model,
        model_name,
        train_data,
        validation_data=None,
        test_data=None,
        resume=False,
        callbacks=None,
    ):
        """Train a model and save results to the trials directory

        Args:
            model: TensorFlow model to train
            model_name: Name of the model
            train_data: Training dataset
            validation_data: Validation dataset (optional)
            test_data: Test dataset (optional)
            resume: Whether to resume training from latest checkpoint if available
            callbacks: Custom callbacks to use during training (optional)

        Returns:
            Tuple of (model, history, metrics)
        """
        # Create run directory in trials folder
        run_dir = self.paths.get_model_trial_dir(model_name)

        # Check for existing checkpoint to resume from
        start_epoch = 0
        if resume:
            checkpoint_dir = Path(run_dir) / "training" / "checkpoints"
            if checkpoint_dir.exists():
                checkpoint_files = list(checkpoint_dir.glob("*.h5"))
                if checkpoint_files:
                    # Find the latest checkpoint
                    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
                    print(f"Resuming training from checkpoint: {latest_checkpoint}")
                    try:
                        model = tf.keras.models.load_model(latest_checkpoint)
                        # Extract epoch number from checkpoint filename if possible
                        try:
                            filename = os.path.basename(latest_checkpoint)
                            epoch_part = filename.split("-")[1]
                            start_epoch = int(epoch_part)
                            print(f"Resuming from epoch {start_epoch}")
                        except:
                            print(
                                "Could not determine start epoch from checkpoint filename"
                            )

                        print("Successfully loaded checkpoint")
                    except Exception as e:
                        print(f"Failed to load checkpoint: {e}")
                        print("Starting fresh training instead")
                        # Proceed with original model if checkpoint loading fails
                else:
                    print(
                        "No checkpoints found to resume from. Starting fresh training."
                    )
            else:
                print("No checkpoint directory found. Starting fresh training.")

        # Initialize training logger
        self.train_logger = Logger(
            f"{model_name}",
            log_dir=run_dir,
            config=self.config.get("logging", {}),
            logger_type="training",
        )
        self.train_logger.log_config(self.config)
        self.train_logger.log_model_summary(model)

        # Initialize separate evaluation logger if configured
        if self.config.get("logging", {}).get("separate_loggers", True):
            self.eval_logger = Logger(
                f"{model_name}",
                log_dir=run_dir,
                config=self.config.get("logging", {}),
                logger_type="evaluation",
            )
        else:
            # Use the same logger for both if separate loggers not configured
            self.eval_logger = self.train_logger

        # Get training parameters from config
        training_config = self.config.get("training", {})
        batch_size = training_config.get("batch_size", 32)
        epochs = training_config.get("epochs", 50)
        learning_rate = training_config.get("learning_rate", 0.001)
        optimizer_name = training_config.get("optimizer", "adam").lower()
        loss = training_config.get("loss", "categorical_crossentropy")
        metrics = training_config.get("metrics", ["accuracy"])

        # Configure mixed precision if enabled
        mixed_precision_enabled = self.config.get("hardware", {}).get(
            "mixed_precision", True
        )
        if mixed_precision_enabled:
            self.train_logger.log_info("Enabling mixed precision training")
            try:
                from tensorflow.keras import mixed_precision

                mixed_precision.set_global_policy("mixed_float16")
                self.train_logger.log_info(
                    "Mixed precision policy set to 'mixed_float16'"
                )
            except Exception as e:
                self.train_logger.log_warning(
                    f"Failed to set mixed precision policy: {e}"
                )

        # Set up optimizer
        if optimizer_name == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

            # Wrap with LossScaleOptimizer for mixed precision if enabled
            if mixed_precision_enabled:
                try:
                    from tensorflow.keras import mixed_precision

                    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
                    self.train_logger.log_info(
                        "Using LossScaleOptimizer wrapper for mixed precision"
                    )
                except Exception as e:
                    self.train_logger.log_warning(
                        f"Failed to create LossScaleOptimizer: {e}"
                    )

        elif optimizer_name == "sgd":
            momentum = training_config.get("momentum", 0.9)
            optimizer = tf.keras.optimizers.SGD(
                learning_rate=learning_rate, momentum=momentum
            )
        elif optimizer_name == "rmsprop":
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        elif optimizer_name == "adagrad":
            optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
        elif optimizer_name == "adadelta":
            optimizer = tf.keras.optimizers.Adadelta(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Add gradient clipping if configured
        clip_norm = training_config.get("clip_norm", None)
        clip_value = training_config.get("clip_value", None)

        # Apply gradient clipping using the version-compatible helper method
        optimizer = self._apply_gradient_clipping(optimizer, clip_norm, clip_value)

        # Compile model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        self.train_logger.log_info(
            f"Model compiled with optimizer: {optimizer_name}, loss: {loss}, metrics: {metrics}"
        )

        # Set up callbacks
        if callbacks is None:
            callbacks = []

        # Get steps per epoch for progress bar
        steps_per_epoch = getattr(train_data, "samples", 0) // batch_size
        setattr(model, "train_step_count", steps_per_epoch)

        # Add progress bar callback
        progress_bar = ProgressBarCallback(epochs=epochs)
        callbacks.append(progress_bar)

        # Model checkpoint callback
        checkpoint_dir = Path(run_dir) / "training" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "checkpoint-{epoch:02d}-{val_loss:.2f}.h5"

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            mode="min",
            save_freq="epoch",
        )
        callbacks.append(checkpoint_callback)

        # TensorBoard callback
        tensorboard_dir = Path(run_dir) / "training" / "tensorboard"
        tensorboard_dir.mkdir(parents=True, exist_ok=True)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=str(tensorboard_dir),
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq="epoch",
        )
        callbacks.append(tensorboard_callback)

        # Early stopping if enabled
        early_stopping_config = training_config.get("early_stopping", {})
        if early_stopping_config.get("enabled", True):
            monitor = early_stopping_config.get("monitor", "val_loss")
            patience = early_stopping_config.get("patience", 10)
            restore_best_weights = early_stopping_config.get(
                "restore_best_weights", True
            )

            early_stopping_callback = tf.keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=restore_best_weights,
                mode="min" if "loss" in monitor else "max",
            )
            callbacks.append(early_stopping_callback)
            self.train_logger.log_info(
                f"Early stopping enabled with patience {patience}, monitoring {monitor}"
            )

        # Learning rate scheduler if enabled (including new warmup schedulers)
        lr_scheduler_config = training_config.get("lr_scheduler", {})
        lr_schedule_config = training_config.get("lr_schedule", {})

        # Check for the new warmup scheduler
        if lr_schedule_config.get("enabled", False):
            # Create a warmup scheduler using the new implementation
            warmup_scheduler = get_warmup_scheduler(self.config)
            if warmup_scheduler:
                callbacks.append(warmup_scheduler)
                scheduler_type = lr_schedule_config.get("type", "warmup_cosine")
                self.train_logger.log_info(
                    f"Using advanced scheduler: {scheduler_type}"
                )

        # Legacy scheduler support
        elif lr_scheduler_config.get("enabled", False):
            lr_scheduler_type = lr_scheduler_config.get("type", "reduce_on_plateau")

            if lr_scheduler_type == "reduce_on_plateau":
                reduce_factor = lr_scheduler_config.get("factor", 0.1)
                reduce_patience = lr_scheduler_config.get("patience", 5)
                reduce_min_lr = lr_scheduler_config.get("min_lr", 1e-6)

                lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=reduce_factor,
                    patience=reduce_patience,
                    min_lr=reduce_min_lr,
                    verbose=1,
                )
                callbacks.append(lr_scheduler)
                self.train_logger.log_info(
                    f"LR scheduler enabled: ReduceLROnPlateau with factor {reduce_factor}"
                )

            elif lr_scheduler_type == "cosine_decay":
                decay_steps = lr_scheduler_config.get("decay_steps", epochs)
                alpha = lr_scheduler_config.get("alpha", 0.0)

                def cosine_decay_schedule(epoch, lr):
                    return learning_rate * (
                        alpha
                        + (1 - alpha) * np.cos(np.pi * epoch / decay_steps) / 2
                        + 0.5
                    )

                lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
                    cosine_decay_schedule
                )
                callbacks.append(lr_scheduler)
                self.train_logger.log_info(
                    f"LR scheduler enabled: Cosine decay over {decay_steps} epochs"
                )

        # Handle class weights if enabled
        class_weights = None
        if training_config.get("class_weight") == "balanced":
            # Get class names
            class_info = getattr(train_data, "class_indices", None)
            if class_info:
                class_names = {v: k for k, v in class_info.items()}
                n_classes = len(class_names)

                try:
                    if hasattr(train_data, "get_files_by_class"):
                        # Direct file counting if available
                        class_counts = [
                            len(list(train_data.get_files_by_class(c)))
                            for c in class_names.values()
                        ]
                        self.train_logger.log_info(
                            f"Class counts from files: {class_counts}"
                        )
                    else:
                        # For tf.data.Dataset and other iterators
                        self.train_logger.log_info(
                            "Dataset doesn't support get_files_by_class, estimating class distribution by sampling"
                        )

                        # Sample more batches for better estimates
                        samples = []
                        max_samples = min(
                            50, len(train_data)
                        )  # Sample more, but not too many

                        # Log the dataset type for debugging
                        self.train_logger.log_info(
                            f"Dataset type: {type(train_data).__name__}"
                        )

                        # Create a temporary copy to avoid consuming the dataset
                        temp_data = train_data
                        if hasattr(train_data, "unbatch"):
                            # If it's a batched dataset, get a sample
                            sample_size = 1000  # Adjust based on your dataset size
                            temp_data = train_data.unbatch().take(sample_size)

                        # Collect samples
                        for i, batch in enumerate(temp_data):
                            if isinstance(batch, tuple):
                                _, y = batch
                            else:
                                y = batch[1]  # Assume second element is labels
                            samples.append(y)
                            if i >= max_samples:
                                break

                        if samples:
                            # Concatenate and calculate class distribution
                            y_samples = np.concatenate(samples, axis=0)
                            if len(y_samples.shape) > 1 and y_samples.shape[1] > 1:
                                # One-hot encoded
                                class_counts = np.sum(y_samples, axis=0).tolist()
                            else:
                                # Integer labels
                                unique, counts = np.unique(
                                    y_samples, return_counts=True
                                )
                                class_counts = [0] * n_classes
                                for cls, count in zip(unique, counts):
                                    class_counts[int(cls)] = count

                            self.train_logger.log_info(
                                f"Estimated class counts: {class_counts}"
                            )
                        else:
                            raise ValueError(
                                "Could not collect samples for class distribution"
                            )

                    # Calculate weights with smoothing to avoid extreme values
                    total = sum(class_counts)
                    smoothing_factor = 0.1
                    class_weights = {
                        i: total
                        / (
                            (n_classes * max(count, 1)) * (1 - smoothing_factor)
                            + (total / n_classes) * smoothing_factor
                        )
                        for i, count in enumerate(class_counts)
                    }

                    self.train_logger.log_info(
                        f"Using balanced class weights: {class_weights}"
                    )
                except Exception as e:
                    self.train_logger.log_warning(
                        f"Failed to compute class weights: {e}. Using uniform weights."
                    )
                    # Log the full exception for debugging
                    import traceback

                    self.train_logger.log_debug(traceback.format_exc())
                    class_weights = None

        # Custom callback to log hardware metrics
        class HardwareMetricsCallback(tf.keras.callbacks.Callback):
            def __init__(self, logger):
                super().__init__()
                self.logger = logger

            def on_epoch_end(self, epoch, logs=None):
                self.logger.log_hardware_metrics(step=epoch)

        callbacks.append(HardwareMetricsCallback(self.train_logger))

        # Train the model
        self.train_logger.log_info(
            f"Starting training for {model_name} with {epochs} epochs"
        )
        start_time = time.time()

        history = model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            initial_epoch=start_epoch,  # Start from the right epoch if resuming
            batch_size=batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0,  # We're using our own progress bar
        )

        training_time = time.time() - start_time
        self.train_logger.log_info(f"Training completed in {training_time:.2f} seconds")

        # Evaluate on test set if provided
        test_metrics = {}
        if test_data is not None:
            self.train_logger.log_info("Evaluating on test data")
            self.eval_logger.log_info("Starting evaluation on test data")
            print("\nEvaluating on test data:")

            # Create a progress bar for evaluation
            test_steps = getattr(test_data, "samples", 0) // batch_size
            with tqdm(total=test_steps, desc="Evaluation") as pbar:
                # Custom callback to update progress bar during evaluation
                class EvalProgressCallback(tf.keras.callbacks.Callback):
                    def on_test_batch_end(self, batch, logs=None):
                        pbar.update(1)
                        if logs and "loss" in logs:
                            pbar.set_postfix(loss=f"{logs['loss']:.4f}")

                test_results = model.evaluate(
                    test_data, verbose=0, callbacks=[EvalProgressCallback()]
                )

            # Create metrics dictionary
            test_metrics = {}
            for i, metric_name in enumerate(model.metrics_names):
                test_metrics[f"test_{metric_name}"] = float(test_results[i])

            # Log test metrics to both loggers
            self.train_logger.log_metrics(test_metrics)
            self.eval_logger.log_metrics(test_metrics)
            self.eval_logger.log_info(
                f"Evaluation completed with accuracy: {test_metrics.get('test_accuracy', 'N/A')}"
            )
        # Combine metrics
        final_metrics = {
            "training_time": training_time,
            "run_dir": str(run_dir),
            **test_metrics,
        }

        # Save history metrics
        for key, values in history.history.items():
            # Save final (last epoch) value
            if values:
                final_metrics[f"final_{key}"] = float(values[-1])
                # Save best value for validation metrics
                if key.startswith("val_"):
                    metric_name = key[4:]  # Remove "val_" prefix
                    if metric_name in ["accuracy", "auc", "precision", "recall"]:
                        # For these metrics, higher is better
                        best_value = max(values)
                        best_epoch = values.index(best_value)
                    else:
                        # For loss and other metrics, lower is better
                        best_value = min(values)
                        best_epoch = values.index(best_value)

                    final_metrics[f"best_{key}"] = float(best_value)
                    final_metrics[f"best_{key}_epoch"] = best_epoch

        # Save final model
        model_path = Path(run_dir) / f"{model_name}_final.h5"
        model.save(str(model_path))
        final_metrics["model_path"] = str(model_path)

        # Save history to CSV
        history_df = pd.DataFrame(history.history)
        history_path = Path(run_dir) / "training" / "history.csv"
        history_df.to_csv(history_path, index=False)

        # Save metrics
        self.train_logger.save_final_metrics(final_metrics)

        # Save evaluation metrics separately if using a different logger
        if self.eval_logger != self.train_logger and test_metrics:
            # Add training time to evaluation metrics
            eval_metrics = {
                "training_time": training_time,
                "run_dir": str(run_dir),
                **test_metrics,
            }
            self.eval_logger.save_final_metrics(eval_metrics)

        # Generate confusion matrix if test data is available
        if test_data is not None and self.config.get("reporting", {}).get(
            "save_confusion_matrix", True
        ):
            try:
                # Get predictions
                self.eval_logger.log_info("Generating evaluation visualizations...")
                y_pred = model.predict(test_data, verbose=0)
                # Get true labels (assuming they're in the second element of the tuple)
                y_true = np.concatenate([y for _, y in test_data], axis=0)

                # Calculate confusion matrix
                y_pred_classes = np.argmax(y_pred, axis=1)
                y_true_classes = np.argmax(y_true, axis=1)

                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(y_true_classes, y_pred_classes)

                # Get class names if available
                class_info = getattr(test_data, "class_indices", None)
                if class_info:
                    class_names = {v: k for k, v in class_info.items()}
                else:
                    class_names = {i: f"Class {i}" for i in range(cm.shape[0])}

                # Log confusion matrix to evaluation logger
                self.eval_logger.log_confusion_matrix(
                    cm,
                    [class_names[i] for i in range(len(class_names))],
                    step=epochs - 1,
                )

                # Calculate additional metrics if requested
                if self.config.get("reporting", {}).get(
                    "save_roc_curves", True
                ) or self.config.get("reporting", {}).get(
                    "save_precision_recall", True
                ):
                    from src.evaluation.metrics import calculate_metrics

                    detailed_metrics = calculate_metrics(
                        y_true, y_pred_classes, y_pred, class_names
                    )

                    # Save detailed metrics
                    metrics_path = (
                        Path(run_dir) / "evaluation" / "detailed_metrics.json"
                    )
                    metrics_path.parent.mkdir(parents=True, exist_ok=True)

                    import json

                    with open(metrics_path, "w") as f:
                        json.dump(detailed_metrics, f, indent=4)

                    # Generate visualization plots
                    from src.evaluation.visualization import (
                        plot_roc_curve,
                        plot_precision_recall_curve,
                        plot_confusion_matrix,
                    )

                    plots_dir = Path(run_dir) / "evaluation" / "plots"
                    plots_dir.mkdir(parents=True, exist_ok=True)

                    if self.config.get("reporting", {}).get(
                        "save_confusion_matrix", True
                    ):
                        cm_path = plots_dir / "confusion_matrix.png"
                        plot_confusion_matrix(
                            y_true, y_pred_classes, class_names, save_path=cm_path
                        )
                        self.eval_logger.log_info(
                            f"Confusion matrix saved to {cm_path}"
                        )

                    if self.config.get("reporting", {}).get("save_roc_curves", True):
                        roc_path = plots_dir / "roc_curve.png"
                        plot_roc_curve(y_true, y_pred, class_names, save_path=roc_path)
                        self.eval_logger.log_info(f"ROC curves saved to {roc_path}")

                    if self.config.get("reporting", {}).get(
                        "save_precision_recall", True
                    ):
                        pr_path = plots_dir / "precision_recall_curve.png"
                        plot_precision_recall_curve(
                            y_true, y_pred, class_names, save_path=pr_path
                        )
                        self.eval_logger.log_info(
                            f"Precision-recall curves saved to {pr_path}"
                        )

            except Exception as e:
                self.eval_logger.log_warning(
                    f"Error generating evaluation visualizations: {e}"
                )
                import traceback

                self.eval_logger.log_debug(traceback.format_exc())

        # Register model in the registry
        try:
            registry = ModelRegistryManager()
            registry.register_model(model, model_name, final_metrics, history, run_dir)
            self.train_logger.log_info(f"Model registered in registry")
        except Exception as e:
            self.train_logger.log_warning(f"Failed to register model in registry: {e}")

        return model, history, final_metrics
```

---

### src/training/training_pipeline.py

```python
"""
Training pipeline module for orchestrating the model training process.
"""

import time
from typing import Dict, List, Any, Optional, Tuple

import tensorflow as tf
import gc

from src.preprocessing.data_loader import DataLoader
from src.models.model_factory import ModelFactory
from src.training.batch_trainer import BatchTrainer
from src.model_registry.registry_manager import ModelRegistryManager


def load_datasets(
    batch_trainer: BatchTrainer,
    config_manager: Any,
    data_loader: DataLoader
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, Dict[int, str]]:
    """
    Load and prepare datasets for training.
    
    Args:
        batch_trainer: BatchTrainer instance for logging
        config_manager: Configuration manager
        data_loader: DataLoader instance
        
    Returns:
        Tuple containing training, validation, test datasets and class names
        
    Raises:
        ValueError: If no classes are found in the dataset
        Exception: For other data loading errors
    """
    batch_trainer.batch_logger.log_info("Loading datasets...")
    
    try:
        # Log data loading strategy
        if config_manager.should_use_tf_data():
            batch_trainer.batch_logger.log_info("Using TensorFlow Data API for dataset loading")
        else:
            batch_trainer.batch_logger.log_info("Using standard data loading pipeline")

        # Load datasets
        train_data, val_data, test_data, class_names = data_loader.load_data(
            config_manager.get_data_directory()
        )

        if not class_names:
            raise ValueError("No classes found in the dataset")

        batch_trainer.batch_logger.log_info(f"Datasets loaded with {len(class_names)} classes")
        batch_trainer.batch_logger.log_info(f"Classes: {list(class_names.values())}")
        
        return train_data, val_data, test_data, class_names
        
    except ValueError as e:
        batch_trainer.batch_logger.log_error(f"Error loading datasets (invalid data): {str(e)}")
        raise
    except Exception as e:
        batch_trainer.batch_logger.log_error(f"Error loading datasets: {str(e)}")
        raise


def execute_training_pipeline(
    config: Dict[str, Any],
    config_manager: Any,
    hardware_info: Dict[str, Any]
) -> Tuple[BatchTrainer, float, int]:
    """
    Execute the full training pipeline.
    
    Args:
        config: Configuration dictionary
        config_manager: Configuration manager
        hardware_info: Hardware configuration information
        
    Returns:
        Tuple containing the batch trainer, total training time, and exit code
        
    Raises:
        Exception: For any training pipeline errors
    """
    # Start timing
    start_time = time.time()
    exit_code = 0
    
    try:
        # Set up batch trainer
        batch_trainer = BatchTrainer(config)
        batch_trainer.setup_batch_logging()

        # Get models to train
        models_to_train = config_manager.get_models_to_train()
        batch_trainer.set_models_to_train(models_to_train)

        # Log hardware configuration
        batch_trainer.batch_logger.log_info(f"Hardware configuration: {hardware_info}")
        batch_trainer.batch_logger.log_hardware_metrics(step=0)
        
        # Choose data loader implementation and load data
        data_loader = DataLoader(config)
        train_data, val_data, test_data, class_names = load_datasets(
            batch_trainer, config_manager, data_loader
        )

        # Initialize model factory
        model_factory = ModelFactory()
        batch_trainer.batch_logger.log_info("Initialized ModelFactory")

        # Initialize model registry
        registry = ModelRegistryManager()

        # Run batch training
        results = batch_trainer.run_batch_training(
            data_loader,
            model_factory,
            train_data,
            val_data,
            test_data,
            class_names,
            resume=config_manager.should_resume_training(),
            attention_type=config_manager.get_attention_type(),
        )

        # Generate comparison report
        batch_trainer.generate_comparison_report()

        # Calculate total time and save summary
        total_time = time.time() - start_time
        batch_trainer.save_batch_summary(total_time)
        
        # Clean up resources to prevent memory leaks
        batch_trainer.cleanup_resources()

    except Exception as e:
        import traceback
        print(f"Error in training pipeline: {str(e)}")
        print(traceback.format_exc())
        
        # Try to clean up resources even on failure
        clean_up_resources()
        
        exit_code = 1
        return None, 0.0, exit_code

    return batch_trainer, time.time() - start_time, exit_code


def generate_training_reports(batch_trainer: BatchTrainer, total_time: float) -> None:
    """
    Generate final reports and print summary for completed training.
    
    Args:
        batch_trainer: BatchTrainer instance
        total_time: Total training time in seconds
    """
    print("\n\nTraining Summary:")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(
        f"Models trained: {batch_trainer.successful_models} successful, "
        f"{batch_trainer.failed_models} failed"
    )
    print("=" * 80)
    print("Training completed.")


def clean_up_resources() -> None:
    """
    Clean up TensorFlow resources and force garbage collection.
    This helps prevent memory leaks between training runs.
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Additional memory cleanup if needed
    if hasattr(tf.keras.backend, 'set_session'):
        tf.keras.backend.set_session(tf.compat.v1.Session())```

---

### src/utils/__init__.py

```python
```

---

### src/utils/cli_utils.py

```python
"""
CLI utilities for handling command-line arguments and configuration loading.
"""

from typing import Tuple, Dict, Any, List, Optional

from src.config.config_manager import ConfigManager


def handle_cli_args() -> Tuple[ConfigManager, Dict[str, Any], bool]:
    """
    Parse command-line arguments and load configuration.
    
    Returns:
        Tuple containing:
            - ConfigManager: Initialized configuration manager
            - Dict: Loaded configuration
            - bool: Whether to print hardware summary and exit
    """
    # Set up configuration manager
    config_manager = ConfigManager()
    args = config_manager.parse_args()
    
    # Check if we should just print hardware summary
    should_print_hardware = config_manager.should_print_hardware_summary()
    
    # Load configuration with command-line overrides
    config = config_manager.load_config()
    
    return config_manager, config, should_print_hardware


def get_project_info(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract project name and version from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple containing project name and version
    """
    project_info = config.get("project", {})
    project_name = project_info.get("name", "Plant Disease Detection")
    project_version = project_info.get("version", "1.0.0")
    
    return project_name, project_version```

---

### src/utils/error_handling.py

```python
"""
Error handling utilities for graceful error handling throughout the codebase.
"""

import sys
import traceback
import logging
from typing import Optional, Callable, Any, Dict, Type, Union
from functools import wraps

# Configure logger
logger = logging.getLogger("plant_disease_detection")


class DataError(Exception):
    """Exception raised for errors in data loading and processing."""
    pass


class ModelError(Exception):
    """Exception raised for errors in model creation and training."""
    pass


class ConfigError(Exception):
    """Exception raised for errors in configuration."""
    pass


def handle_exception(
    exc: Exception, 
    error_msg: str, 
    log_traceback: bool = True
) -> None:
    """
    Handle exceptions with consistent logging and user feedback.
    
    Args:
        exc: The exception that was raised
        error_msg: A human-readable error message
        log_traceback: Whether to log the full traceback
    """
    # Log the error with appropriate level
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        logger.error(f"{error_msg}: {str(exc)}")
    else:
        logger.critical(f"{error_msg}: {str(exc)}")
    
    # Log traceback for unexpected errors
    if log_traceback:
        logger.debug(traceback.format_exc())
    
    # Print user-friendly message to console
    print(f"Error: {error_msg}")
    print(f"Details: {str(exc)}")


def try_except_decorator(
    error_msg: str,
    exception_types: Union[Type[Exception], tuple] = Exception,
    cleanup_func: Optional[Callable] = None
) -> Callable:
    """
    Decorator for try-except error handling pattern.
    
    Args:
        error_msg: Message to show when an exception occurs
        exception_types: Type(s) of exceptions to catch
        cleanup_func: Optional function to call in finally block
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                handle_exception(e, error_msg)
                raise
            finally:
                if cleanup_func:
                    cleanup_func()
        return wrapper
    return decorator


def retry_on_error(
    max_retries: int = 3,
    retry_exceptions: Union[Type[Exception], tuple] = Exception,
    backoff_factor: float = 2.0
) -> Callable:
    """
    Decorator that retries a function on specific exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_exceptions: Exception types to retry on
        backoff_factor: Factor to multiply delay between retries
        
    Returns:
        Decorated function with retry capabilities
    """
    import time
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = 1.0
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) reached for {func.__name__}")
                        raise
                    
                    wait_time = delay * (backoff_factor ** (retries - 1))
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.1f}s due to: {str(e)}"
                    )
                    time.sleep(wait_time)
            
            # This should never be reached
            raise RuntimeError("Unexpected end of retry loop")
            
        return wrapper
    return decorator```

---

### src/utils/hardware_utils.py

```python
"""
Hardware configuration utilities for TensorFlow setup.

This module provides functions to configure TensorFlow for optimal 
performance on different hardware platforms (CPU, GPU, Apple Silicon).
"""

import platform
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)


def configure_hardware(config):
    """Configure TensorFlow for hardware acceleration

    This function configures TensorFlow based on the available hardware and
    the provided configuration. It handles CPU threading, GPU memory growth,
    Apple Silicon Metal support, and mixed precision training.

    Args:
        config: Configuration dictionary with hardware settings

    Returns:
        Dictionary with hardware configuration information
    """
    hardware_info = {
        "platform": platform.system(),
        "cpu_type": platform.machine(),
        "tensorflow_version": tf.__version__,
        "gpu_available": len(tf.config.list_physical_devices("GPU")) > 0,
        "devices_used": [],
    }

    hardware_config = config.get("hardware", {})

    # Set threading parameters FIRST (before any TF operations)
    try:
        tf.config.threading.set_inter_op_parallelism_threads(
            hardware_config.get("inter_op_parallelism", 0)
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            hardware_config.get("intra_op_parallelism", 0)
        )
        print("Threading parameters configured successfully")
    except RuntimeError as e:
        print(f"Warning: Could not set threading parameters: {e}")

    # Detect Apple Silicon
    is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    hardware_info["is_apple_silicon"] = is_apple_silicon

    # Configure TensorFlow for Metal on Apple Silicon
    if (
        hardware_config.get("use_metal", True)
        and is_apple_silicon
        and hardware_info["gpu_available"]
    ):
        print("Configuring TensorFlow for Metal on Apple Silicon")
        hardware_info["using_metal"] = True

        # Enable Metal
        try:
            gpu_devices = tf.config.list_physical_devices("GPU")
            if gpu_devices:
                tf.config.experimental.set_visible_devices(gpu_devices[0], "GPU")
                hardware_info["devices_used"].append("Metal GPU")

                # Enable memory growth to prevent allocating all GPU memory at once
                if hardware_config.get("memory_growth", True):
                    for gpu in gpu_devices:
                        try:
                            tf.config.experimental.set_memory_growth(gpu, True)
                            print(f"Enabled memory growth for {gpu}")
                        except Exception as e:
                            print(
                                f"Warning: Could not set memory growth for {gpu}: {e}"
                            )

                # Use mixed precision if enabled
                if hardware_config.get("mixed_precision", True):
                    try:
                        tf.keras.mixed_precision.set_global_policy("mixed_float16")
                        print("Mixed precision enabled (float16)")
                        hardware_info["mixed_precision"] = True
                    except Exception as e:
                        print(f"Warning: Could not set mixed precision: {e}")
                        hardware_info["mixed_precision"] = False
        except Exception as e:
            print(f"Warning: Error configuring Metal: {e}")
            hardware_info["error_configuring_metal"] = str(e)
            hardware_info["using_metal"] = False

    # For CUDA GPUs
    elif hardware_info["gpu_available"] and hardware_config.get("use_gpu", True):
        print("Configuring TensorFlow for CUDA GPU")
        hardware_info["using_cuda"] = True

        try:
            # Get GPU details
            gpu_devices = tf.config.list_physical_devices("GPU")
            for i, gpu in enumerate(gpu_devices):
                hardware_info["devices_used"].append(
                    f"CUDA GPU {i}: {gpu.name if hasattr(gpu, 'name') else 'Unknown'}"
                )

            # Enable memory growth to prevent allocating all GPU memory at once
            if hardware_config.get("memory_growth", True):
                for gpu in gpu_devices:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                        print(f"Enabled memory growth for {gpu}")
                    except Exception as e:
                        print(f"Warning: Could not set memory growth for {gpu}: {e}")

            # Use mixed precision if enabled
            if hardware_config.get("mixed_precision", True):
                try:
                    tf.keras.mixed_precision.set_global_policy("mixed_float16")
                    print("Mixed precision enabled (float16)")
                    hardware_info["mixed_precision"] = True
                except Exception as e:
                    print(f"Warning: Could not set mixed precision: {e}")
                    hardware_info["mixed_precision"] = False
        except Exception as e:
            print(f"Warning: Error configuring GPU: {e}")
            hardware_info["error_configuring_gpu"] = str(e)
    else:
        print("Using CPU for computation")
        hardware_info["using_cpu"] = True
        hardware_info["devices_used"].append("CPU")

    # Set memory limit if specified
    memory_limit_mb = hardware_config.get("memory_limit_mb")
    if memory_limit_mb:
        try:
            for gpu in tf.config.list_physical_devices("GPU"):
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=memory_limit_mb
                        )
                    ],
                )
            print(f"GPU memory limit set to {memory_limit_mb}MB")
            hardware_info["memory_limit_mb"] = memory_limit_mb
        except Exception as e:
            print(f"Warning: Could not set memory limit: {e}")

    return hardware_info


def print_hardware_summary():
    """Print a summary of available hardware for TensorFlow"""

    print("\n=== Hardware Summary ===")
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Processor: {platform.processor()}")
    print(f"Python version: {platform.python_version()}")
    print(f"TensorFlow version: {tf.__version__}")

    # Check for GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\nGPUs Available: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name if hasattr(gpu, 'name') else gpu}")

            # Try to get memory info
            try:
                gpu_details = tf.config.experimental.get_device_details(gpu)
                if gpu_details and "memory_limit" in gpu_details:
                    mem_gb = round(gpu_details["memory_limit"] / (1024**3), 2)
                    print(f"    Memory: {mem_gb} GB")
            except:
                pass
    else:
        print("\nNo GPUs available")

    # Check if using Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("\nApple Silicon detected")
        print("  Metal support is available for TensorFlow acceleration")

    print("\nCPU Information:")
    print(f"  Logical CPUs: {tf.config.threading.get_inter_op_parallelism_threads()}")

    # Check if mixed precision is available
    try:
        policy = tf.keras.mixed_precision.global_policy()
        print(f"\nCurrent precision policy: {policy.name}")
    except:
        print("\nMixed precision status: Unknown")

    print("======================\n")


def get_optimal_batch_size(
    model, starting_batch_size=32, target_memory_usage=0.7, max_attempts=5
):
    """Estimate an optimal batch size for a model based on memory constraints

    This is an experimental function that tries to find a batch size that
    uses a target fraction of available GPU memory.

    Args:
        model: A TensorFlow model
        starting_batch_size: Initial batch size to try
        target_memory_usage: Target fraction of memory to utilize (0.0-1.0)
        max_attempts: Maximum number of batch size adjustments to try

    Returns:
        Estimated optimal batch size or the starting_batch_size if estimation fails
    """
    # Ensure we have GPU available, otherwise return the starting batch size
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU available, using default batch size")
        return starting_batch_size

    # Get input shape from the model
    if hasattr(model, "input_shape"):
        input_shape = model.input_shape
        if isinstance(input_shape, tuple) and None in input_shape:
            # Replace None with a reasonable value (batch dimension)
            input_shape = list(input_shape)
            input_shape[0] = starting_batch_size
            input_shape = tuple(input_shape)
    else:
        print("Could not determine model input shape, using default batch size")
        return starting_batch_size

    try:
        # Try to estimate appropriate batch size
        current_batch_size = starting_batch_size

        for _ in range(max_attempts):
            # Create a test batch
            test_batch = tf.random.normal(input_shape)

            # Run a forward pass
            with tf.GradientTape() as tape:
                _ = model(test_batch, training=True)

            # Try to get memory info
            try:
                memory_info = tf.config.experimental.get_memory_info("GPU:0")
                if memory_info:
                    current_usage = memory_info["current"] / memory_info["peak"]
                    if current_usage < target_memory_usage * 0.8:
                        # Increase batch size
                        current_batch_size = int(current_batch_size * 1.5)
                    elif current_usage > target_memory_usage * 1.1:
                        # Decrease batch size
                        current_batch_size = int(current_batch_size * 0.7)
                    else:
                        # Good batch size found
                        break
            except:
                # If memory info not available, make a conservative guess
                break

        print(f"Estimated optimal batch size: {current_batch_size}")
        return current_batch_size

    except Exception as e:
        print(f"Error estimating optimal batch size: {e}")
        return starting_batch_size
```

---

### src/utils/logger.py

```python
import os
import logging
import json
import time
import platform
from pathlib import Path
import tensorflow as tf
from datetime import datetime
from tqdm.auto import tqdm
import numpy as np

# Import psutil conditionally for hardware monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from src.config.config import get_paths


class Logger:
    def __init__(self, name, log_dir=None, config=None, logger_type="training"):
        """Initialize the logging system.

        Args:
            name: Name of the logger (e.g., model name or experiment name)
            log_dir: Directory to save logs. If None, uses the trials directory.
            config: Configuration dictionary for logging settings.
            logger_type: Type of logger - "training" or "evaluation"
        """
        self.name = name
        self.config = config or {}
        self.paths = get_paths()
        self.logger_type = logger_type

        # Set up log directory
        if log_dir is None:
            # If no log_dir is provided, use trials directory
            # This shouldn't happen with our configuration
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = self.paths.trials_dir / f"{name}_{timestamp}"
        else:
            # Use the provided directory
            log_dir_path = Path(log_dir)

            # Create subdirectory for logger type
            if logger_type == "training":
                self.log_dir = log_dir_path / "training"
            elif logger_type == "evaluation":
                self.log_dir = log_dir_path / "evaluation"
            else:
                # Default to root of log_dir if type is unknown
                self.log_dir = log_dir_path

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Set up logging
        self.logger = logging.getLogger(f"{name}_{logger_type}")
        self.logger.setLevel(self._get_log_level())

        # Clear any existing handlers to avoid duplicates
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Set up file handler
        log_file = Path(self.log_dir / f"{name}_{logger_type}.log")
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Set up console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # Set up TensorBoard if enabled
        self.tensorboard_writer = None
        if self.config.get("logging", {}).get("tensorboard", True):
            tensorboard_dir = Path(self.log_dir / "tensorboard")
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard_writer = tf.summary.create_file_writer(
                str(tensorboard_dir)
            )

        # Initialize metrics tracking
        self.metrics = {}
        self.start_time = time.time()

        # Log system info (only for training logger to avoid duplication)
        if logger_type == "training":
            self._log_system_info()

        self.logger.info(
            f"{logger_type.capitalize()} logger initialized. Logs will be saved to {self.log_dir}"
        )

    def _get_log_level(self):
        """Get log level from config or default to INFO"""
        level_str = self.config.get("logging", {}).get("level", "INFO").upper()
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        return level_map.get(level_str, logging.INFO)

    def _log_system_info(self):
        """Log information about the system and environment"""
        # System info
        system_info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "processor": platform.processor(),
            "tensorflow_version": tf.__version__,
            "timestamp": datetime.now().isoformat(),
            "logger_type": self.logger_type,
        }

        # Add more detailed system info if psutil is available
        if PSUTIL_AVAILABLE:
            system_info.update(
                {
                    "cpu_count": psutil.cpu_count(logical=False),
                    "logical_cpus": psutil.cpu_count(logical=True),
                    "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                }
            )

        # Check for GPU/Metal
        system_info["gpu_available"] = len(tf.config.list_physical_devices("GPU")) > 0
        system_info["metal_available"] = (
            platform.system() == "Darwin" and platform.machine() == "arm64"
        )

        # Get GPU details
        devices = []
        for i, gpu in enumerate(tf.config.list_physical_devices("GPU")):
            devices.append(f"GPU {i}: {gpu.name}")

        # For Apple Silicon
        if system_info["metal_available"]:
            devices.append("Metal: Apple Silicon")

        system_info["devices"] = devices

        # Get more detailed GPU info if available
        try:
            for i, gpu in enumerate(tf.config.list_physical_devices("GPU")):
                # Try to get memory limit if set
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    if gpu_details and "memory_limit" in gpu_details:
                        mem_gb = round(gpu_details["memory_limit"] / (1024**3), 2)
                        system_info[f"gpu_{i}_memory_limit_gb"] = mem_gb
                except:
                    pass
        except:
            pass

        # Log to file and console
        self.logger.info(f"System Info: {json.dumps(system_info, indent=2)}")

        # Save as JSON
        system_info_path = Path(self.log_dir / "system_info.json")
        with open(system_info_path, "w") as f:
            json.dump(system_info, f, indent=4)

    def log_info(self, message):
        """Log an info message"""
        self.logger.info(message)

    def log_warning(self, message):
        """Log a warning message"""
        self.logger.warning(message)

    def log_error(self, message):
        """Log an error message"""
        self.logger.error(message)

    def log_debug(self, message):
        """Log a debug message"""
        self.logger.debug(message)

    def log_config(self, config):
        """Log the configuration used for training"""
        # Create a clean copy of the config that's JSON-serializable
        clean_config = {}
        for key, value in config.items():
            if isinstance(value, dict):
                clean_config[key] = {}
                for k, v in value.items():
                    if isinstance(v, (str, int, float, bool, list, dict)) or v is None:
                        clean_config[key][k] = v
            elif isinstance(value, (str, int, float, bool, list)) or value is None:
                clean_config[key] = value

        self.logger.info(f"Configuration: {json.dumps(clean_config, indent=2)}")

        # Save config as JSON
        config_path = Path(self.log_dir / "config.json")
        with open(config_path, "w") as f:
            json.dump(clean_config, f, indent=4)

    def log_model_summary(self, model):
        """Log model architecture summary"""
        # Create a string buffer to capture the summary
        model_summary = []
        model.summary(print_fn=lambda x: model_summary.append(x))

        # Log to file
        summary_path = Path(self.log_dir / "model_summary.txt")
        with open(summary_path, "w") as f:
            f.write("\n".join(model_summary) + "\n")

        self.logger.info(f"Model summary saved to {summary_path}")

        # Try to generate a model diagram
        try:
            if len(model.layers) <= 100:  # Skip for very complex models
                dot_img_file = Path(self.log_dir / "model_diagram.png")
                tf.keras.utils.plot_model(
                    model, to_file=dot_img_file, show_shapes=True, show_layer_names=True
                )
                self.logger.info(f"Model diagram saved to {dot_img_file}")
        except Exception as e:
            self.logger.debug(f"Could not generate model diagram: {e}")

    def log_metrics(self, metrics, step=None):
        """Log metrics during training or evaluation"""
        # Update metrics dictionary
        for key, value in metrics.items():
            # Convert TensorFlow tensors to Python types
            if hasattr(value, "numpy"):
                try:
                    value = value.numpy()
                except:
                    pass

            # Convert NumPy values to Python types
            if hasattr(np, "float32") and isinstance(
                value, (np.float32, np.float64, np.int32, np.int64)
            ):
                value = value.item()

            self.metrics[key] = value

        # Log to console and file
        metrics_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
        )
        self.logger.info(f"Metrics - {metrics_str}")

        # Log to TensorBoard if enabled
        if self.tensorboard_writer and step is not None:
            with self.tensorboard_writer.as_default():
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        tf.summary.scalar(key, value, step=step)
                self.tensorboard_writer.flush()

    def log_hardware_metrics(self, step=None):
        """Log hardware utilization metrics including GPU/Metal activity detection"""
        if not PSUTIL_AVAILABLE:
            self.logger.warning(
                "psutil not installed; hardware metrics logging disabled"
            )
            return {}

        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            per_cpu = psutil.cpu_percent(interval=0.1, percpu=True)

            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)

            # GPU usage tracking
            gpu_info = {}
            gpu_active = False

            # Check if GPU devices are available
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                gpu_info["available"] = len(gpus)

                # Try to detect if GPU is being used
                try:
                    # For Metal on Apple Silicon
                    if platform.system() == "Darwin" and platform.machine() == "arm64":
                        # Simple test tensor operation on GPU to check if it's working
                        with tf.device("/GPU:0"):
                            # Create and immediately use a test tensor
                            a = tf.random.normal([1000, 1000])
                            b = tf.random.normal([1000, 1000])
                            c = tf.matmul(a, b)  # Matrix multiplication to engage GPU
                            # Force execution
                            _ = c.numpy()

                            # Check if operation was actually done on GPU
                            gpu_active = True
                            gpu_info["active"] = True

                        # Try to get memory usage (experimental API)
                        try:
                            memory_info = tf.config.experimental.get_memory_info(
                                "/device:GPU:0"
                            )
                            if memory_info:
                                gpu_info["memory_used_mb"] = memory_info.get(
                                    "current", 0
                                ) / (1024**2)
                                gpu_info["memory_peak_mb"] = memory_info.get(
                                    "peak", 0
                                ) / (1024**2)
                        except:
                            pass
                except Exception as e:
                    self.logger.debug(f"Error checking GPU activity: {e}")
                    gpu_info["error"] = str(e)

            # Format hardware metrics string
            hw_metrics_str = f"Hardware - CPU: {cpu_percent}%, Memory: {memory_percent}% ({memory_used_gb:.2f} GB)"

            # Add GPU info if available
            if gpus:
                gpu_status = "ACTIVE" if gpu_active else "INACTIVE"
                hw_metrics_str += f", GPU: {gpu_status}"

                # Add memory info if available
                if "memory_used_mb" in gpu_info:
                    hw_metrics_str += f" (Memory: {gpu_info['memory_used_mb']:.2f} MB)"

            # Log to console and file
            self.logger.info(hw_metrics_str)

            # Create hardware metrics dict for tensorboard
            hw_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory_used_gb,
                "gpu_active": 1 if gpu_active else 0,
            }

            # Add per-CPU metrics
            for i, cpu in enumerate(per_cpu):
                hw_metrics[f"cpu_{i}_percent"] = cpu

            # Add GPU metrics
            if "memory_used_mb" in gpu_info:
                hw_metrics["gpu_memory_used_mb"] = gpu_info["memory_used_mb"]
                hw_metrics["gpu_memory_peak_mb"] = gpu_info.get("memory_peak_mb", 0)

            # Log to TensorBoard if enabled
            if self.tensorboard_writer and step is not None:
                with self.tensorboard_writer.as_default():
                    for key, value in hw_metrics.items():
                        tf.summary.scalar(f"hardware/{key}", value, step=step)
                    self.tensorboard_writer.flush()

            return hw_metrics

        except Exception as e:
            self.logger.warning(f"Error in hardware metrics logging: {str(e)}")
            return {}

    def log_training_progress(self, epoch, batch, metrics, total_batches):
        """Log progress during training with tqdm progress bar"""
        # Calculate time and ETA
        time_elapsed = time.time() - self.start_time
        progress = (batch + 1) / total_batches
        eta = time_elapsed / (progress + 1e-8) * (1 - progress)

        # Format metrics for logging
        metrics_str = ", ".join(
            [
                f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                for k, v in metrics.items()
            ]
        )

        # Log to file
        self.logger.info(
            f"Epoch {epoch+1} - Batch {batch+1}/{total_batches} - {metrics_str} - "
            f"ETA: {eta:.2f}s"
        )

        # For tqdm integration, we would typically use the ProgressBarCallback class
        # which is managed by the Trainer class

    def log_images(self, images, step, name="images"):
        """Log images to TensorBoard

        Args:
            images: Batch of images (shape [N, H, W, C])
            step: Current step
            name: Name for the images
        """
        if self.tensorboard_writer:
            with self.tensorboard_writer.as_default():
                tf.summary.image(name, images, step=step, max_outputs=10)
                self.tensorboard_writer.flush()

    def log_confusion_matrix(self, cm, class_names, step):
        """Log confusion matrix as an image to TensorBoard

        Args:
            cm: Confusion matrix (shape [num_classes, num_classes])
            class_names: List of class names
            step: Current step
        """
        if self.tensorboard_writer:
            try:
                import matplotlib.pyplot as plt
                import io

                # Create figure and plot confusion matrix
                figure = plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title("Confusion Matrix")
                plt.colorbar()
                tick_marks = range(len(class_names))
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)

                # Label the matrix
                thresh = cm.max() / 2.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j,
                            i,
                            format(cm[i, j], "d"),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                        )

                plt.tight_layout()
                plt.ylabel("True label")
                plt.xlabel("Predicted label")

                # Convert figure to image
                buf = io.BytesIO()
                plt.savefig(buf, format="png")
                plt.close(figure)
                buf.seek(0)
                image = tf.image.decode_png(buf.getvalue(), channels=4)
                image = tf.expand_dims(image, 0)

                # Log to TensorBoard
                with self.tensorboard_writer.as_default():
                    tf.summary.image("confusion_matrix", image, step=step)
                    self.tensorboard_writer.flush()

                # Also save the confusion matrix as an image file
                cm_path = Path(self.log_dir / f"confusion_matrix_epoch_{step}.png")
                plt.figure(figsize=(10, 8))
                plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
                plt.title("Confusion Matrix")
                plt.colorbar()
                plt.xticks(tick_marks, class_names, rotation=45)
                plt.yticks(tick_marks, class_names)

                # Label the matrix
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(
                            j,
                            i,
                            format(cm[i, j], "d"),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                        )

                plt.tight_layout()
                plt.ylabel("True label")
                plt.xlabel("Predicted label")
                plt.savefig(cm_path)
                plt.close()

                self.logger.info(f"Confusion matrix saved to {cm_path}")

            except Exception as e:
                self.logger.warning(f"Error logging confusion matrix: {e}")

    def save_final_metrics(self, metrics):
        """Save final metrics at the end of training or evaluation"""
        # Update metrics dictionary
        for key, value in metrics.items():
            # Convert TensorFlow tensors to Python types
            if hasattr(value, "numpy"):
                try:
                    value = value.numpy()
                    if hasattr(value, "item"):
                        value = value.item()
                except:
                    value = str(value)

            # Convert NumPy values to Python types
            if hasattr(np, "float32") and isinstance(
                value, (np.float32, np.float64, np.int32, np.int64)
            ):
                value = value.item()

            self.metrics[key] = value

        # Add timing information
        self.metrics["training_time_seconds"] = time.time() - self.start_time
        self.metrics["training_time_human"] = self._format_time(
            self.metrics["training_time_seconds"]
        )
        self.metrics["timestamp_end"] = datetime.now().isoformat()
        self.metrics["logger_type"] = self.logger_type

        # First save to the log directory
        log_metrics_path = Path(self.log_dir / f"final_metrics_{self.logger_type}.json")
        with open(log_metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        # Also save to the parent directory (for the model registry)
        # Only if this is a training logger
        if self.logger_type == "training":
            parent_metrics_path = Path(self.log_dir).parent / "final_metrics.json"
            with open(parent_metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=4)
            self.logger.info(
                f"Final metrics saved to {log_metrics_path} and {parent_metrics_path}"
            )
        else:
            self.logger.info(f"Final metrics saved to {log_metrics_path}")

        return self.metrics

    def _format_time(self, seconds):
        """Format time in seconds to a human-readable string"""
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if hours > 0:
            parts.append(f"{int(hours)}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{int(minutes)}m")
        parts.append(f"{int(seconds)}s")

        return " ".join(parts)


class ProgressBarManager:
    """A class to manage tqdm progress bars for training and evaluation"""

    def __init__(self, total=None, desc=None, position=0, leave=True):
        """Initialize a progress bar manager

        Args:
            total: Total number of items
            desc: Description for the progress bar
            position: Position of the progress bar (for nested bars)
            leave: Whether to leave the progress bar after completion
        """
        self.total = total
        self.desc = desc
        self.position = position
        self.leave = leave
        self.pbar = None

    def __enter__(self):
        """Create and return the progress bar when entering a context"""
        self.pbar = tqdm(
            total=self.total, desc=self.desc, position=self.position, leave=self.leave
        )
        return self.pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close the progress bar when exiting the context"""
        if self.pbar is not None:
            self.pbar.close()
            self.pbar = None
```

---

### src/utils/memory_utils.py

```python
"""
Memory management utilities to prevent memory leaks during training.
"""

import gc
import os
import sys
import psutil
import logging
from typing import Dict, Any, Optional, Callable

import tensorflow as tf
import numpy as np

logger = logging.getLogger("plant_disease_detection")


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage statistics
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    memory_stats = {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        "percent_used": process.memory_percent(),
        "system_total_gb": system_memory.total / (1024 * 1024 * 1024),
        "system_available_gb": system_memory.available / (1024 * 1024 * 1024),
        "system_percent": system_memory.percent
    }
    
    return memory_stats


def log_memory_usage(step: int = 0, prefix: str = "") -> Dict[str, Any]:
    """
    Log current memory usage.
    
    Args:
        step: Current step (for logging)
        prefix: Prefix for log message
        
    Returns:
        Memory usage statistics
    """
    memory_stats = get_memory_usage()
    
    log_message = f"{prefix}Memory usage: "
    log_message += f"RSS: {memory_stats['rss_mb']:.1f}MB, "
    log_message += f"Process: {memory_stats['percent_used']:.1f}%, "
    log_message += f"System: {memory_stats['system_percent']:.1f}%"
    
    logger.info(log_message)
    return memory_stats


def clean_memory(clean_gpu: bool = True) -> None:
    """
    Clean up memory resources and force garbage collection.
    
    Args:
        clean_gpu: Whether to clear GPU memory as well
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Additional TensorFlow cleanup
    if hasattr(tf.keras.backend, 'set_session'):
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
    
    # Clean GPU memory if requested and available
    if clean_gpu and tf.config.list_physical_devices('GPU'):
        try:
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.reset_memory_stats(gpu)
        except Exception as e:
            logger.warning(f"Failed to reset GPU memory stats: {e}")


def memory_monitoring_decorator(
    func: Callable,
    log_prefix: str = "",
    log_interval: int = 1
) -> Callable:
    """
    Decorator to monitor memory usage during a function execution.
    
    Args:
        func: Function to decorate
        log_prefix: Prefix for log messages
        log_interval: Interval for memory logging (in seconds)
        
    Returns:
        Decorated function
    """
    from functools import wraps
    import time
    import threading
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        stop_monitor = threading.Event()
        memory_stats = []
        
        def monitor_memory():
            start_time = time.time()
            while not stop_monitor.is_set():
                try:
                    stats = get_memory_usage()
                    stats["time"] = time.time() - start_time
                    memory_stats.append(stats)
                    
                    # Log current memory usage
                    log_memory_usage(step=len(memory_stats), prefix=log_prefix)
                    
                    # Wait for next interval
                    time.sleep(log_interval)
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")
                    break
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop monitoring
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)
            
            # Log final memory usage
            final_stats = log_memory_usage(prefix=f"{log_prefix}Final ")
            
            # Clean up resources
            clean_memory()
    
    return wrapper


def limit_gpu_memory_growth() -> None:
    """
    Configure TensorFlow to limit GPU memory growth to prevent OOM errors.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            logger.info(f"GPU memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            logger.error(f"Error setting GPU memory growth: {e}")```

---

### src/utils/report_generator.py

```python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import datetime
from jinja2 import Template


from src.config.config import get_paths


class ReportGenerator:
    def __init__(self, config=None):
        self.config = config or {}
        self.paths = get_paths()

    def generate_model_report(
        self, model_name, run_dir, metrics, history=None, class_names=None
    ):
        """Generate an HTML report for a model run"""
        # Convert run_dir to Path if it's a string
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        # Load metrics
        if isinstance(metrics, str) and os.path.exists(metrics):
            with open(metrics, "r") as f:
                metrics = json.load(f)

        # Load history if path provided
        if isinstance(history, str) and os.path.exists(history):
            history_df = pd.read_csv(history)
            history_dict = {col: history_df[col].tolist() for col in history_df.columns}
            history = type("obj", (object,), {"history": history_dict})

        # Create plots directory if needed
        plots_dir = run_dir / "training" / "plots"
        os.makedirs(plots_dir, exist_ok=True)

        # Generate plots if history is available
        if history is not None and hasattr(history, "history"):
            # Import here to avoid circular imports
            from src.evaluation.visualization import plot_training_history

            history_plot_path = plots_dir / "training_history.png"
            plot_training_history(history, save_path=history_plot_path)

            # Create additional plots if enabled in config
            if self.config.get("reporting", {}).get("generate_plots", True):
                # Generate learning rate plot if available
                if "lr" in history.history:
                    self._plot_learning_rate(history, plots_dir / "learning_rate.png")

                # Generate loss and metrics comparison plots
                metrics_to_plot = [
                    k
                    for k in history.history.keys()
                    if not k.startswith("lr") and not k.startswith("val_")
                ]

                for metric in metrics_to_plot:
                    val_metric = f"val_{metric}"
                    if val_metric in history.history:
                        self._plot_metric_comparison(
                            history,
                            metric,
                            val_metric,
                            plots_dir / f"{metric}_comparison.png",
                        )

        # Create report context
        context = {
            "model_name": model_name,
            "run_dir": str(run_dir),
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
            "has_history": history is not None and hasattr(history, "history"),
            "class_names": class_names,
            "project_name": self.config.get("project", {}).get(
                "name", "Plant Disease Detection"
            ),
            "project_version": self.config.get("project", {}).get("version", "1.0.0"),
        }

        # Generate HTML report using template
        html = self._render_report_template(context)

        # Save report to file
        report_path = run_dir / "report.html"
        with open(report_path, "w") as f:
            f.write(html)

        return report_path

    def _plot_learning_rate(self, history, save_path):
        """Plot learning rate over epochs"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history["lr"])
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _plot_metric_comparison(self, history, train_metric, val_metric, save_path):
        """Plot comparison between training and validation metrics"""
        plt.figure(figsize=(10, 5))
        plt.plot(history.history[train_metric], label=f"Training {train_metric}")
        plt.plot(history.history[val_metric], label=f"Validation {val_metric}")
        plt.title(f"{train_metric} Comparison")
        plt.xlabel("Epoch")
        plt.ylabel(train_metric)
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def _render_report_template(self, context):
        """Render HTML report using Jinja2 template"""
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>{{ model_name }} Training Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    color: #333;
                    background-color: #f8f9fa;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .card {
                    background: #fff;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    padding: 20px;
                    margin-bottom: 20px;
                }
                .header {
                    background-color: #2c3e50;
                    color: white;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }
                .header h1 {
                    color: white;
                    margin: 0;
                }
                .header p {
                    margin: 5px 0 0 0;
                    opacity: 0.8;
                }
                table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }
                th, td {
                    padding: 12px 15px;
                    text-align: left;
                    border-bottom: 1px solid #ddd;
                }
                th {
                    background-color: #f8f9fa;
                }
                .metric-value {
                    font-weight: bold;
                }
                .good-metric {
                    color: #28a745;
                }
                .average-metric {
                    color: #fd7e14;
                }
                .poor-metric {
                    color: #dc3545;
                }
                .plot-container {
                    text-align: center;
                    margin: 20px 0;
                }
                .plot-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 5px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                }
                .plot-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 20px;
                    margin-top: 20px;
                }
                .footer {
                    margin-top: 30px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    padding: 20px;
                    border-top: 1px solid #eee;
                }
                .summary {
                    font-size: 1.2em;
                    margin: 20px 0;
                    padding: 15px;
                    background-color: #f1f8ff;
                    border-left: 4px solid #4285f4;
                    border-radius: 3px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{{ model_name }} Training Report</h1>
                    <p>{{ project_name }} v{{ project_version }} | Generated on {{ generation_time }}</p>
                </div>
                
                <div class="card">
                    <h2>Model Information</h2>
                    <table>
                        <tr>
                            <th>Model Name</th>
                            <td>{{ model_name }}</td>
                        </tr>
                        <tr>
                            <th>Run Directory</th>
                            <td>{{ run_dir }}</td>
                        </tr>
                        {% if "training_time" in metrics %}
                        <tr>
                            <th>Training Time</th>
                            <td>{{ "%.2f"|format(metrics["training_time"]) }} seconds ({{ "%.2f"|format(metrics["training_time"]/60) }} minutes)</td>
                        </tr>
                        {% endif %}
                        {% if class_names %}
                        <tr>
                            <th>Classes</th>
                            <td>{{ class_names|length }} classes
                                {% if class_names|length <= 20 %}
                                    <br><small>{{ class_names|join(', ') }}</small>
                                {% endif %}
                            </td>
                        </tr>
                        {% endif %}
                    </table>
                </div>
                
                {% if "test_accuracy" in metrics or "val_accuracy" in metrics %}
                <div class="summary">
                    Model Performance Summary: 
                    {% if "test_accuracy" in metrics %}
                        Test Accuracy: <span class="metric-value 
                            {% if metrics["test_accuracy"] > 0.9 %}good-metric
                            {% elif metrics["test_accuracy"] > 0.8 %}average-metric
                            {% else %}poor-metric{% endif %}">
                            {{ "%.2f"|format(metrics["test_accuracy"] * 100) }}%
                        </span>
                    {% elif "val_accuracy" in metrics %}
                        Validation Accuracy: <span class="metric-value 
                            {% if metrics["val_accuracy"] > 0.9 %}good-metric
                            {% elif metrics["val_accuracy"] > 0.8 %}average-metric
                            {% else %}poor-metric{% endif %}">
                            {{ "%.2f"|format(metrics["val_accuracy"] * 100) }}%
                        </span>
                    {% endif %}
                </div>
                {% endif %}
                
                <div class="card">
                    <h2>Performance Metrics</h2>
                    <table>
                        {% for key, value in metrics.items() %}
                        <tr>
                            <th>{{ key }}</th>
                            <td class="metric-value">
                                {% if value is number %}
                                    {{ "%.4f"|format(value) }}
                                    {% if "accuracy" in key or "precision" in key or "recall" in key or "f1" in key or "auc" in key %}
                                        ({{ "%.2f"|format(value * 100) }}%)
                                    {% endif %}
                                {% else %}
                                    {{ value }}
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>
                
                {% if has_history %}
                <div class="card">
                    <h2>Training Visualization</h2>
                    <div class="plot-container">
                        <img src="training/plots/training_history.png" alt="Training History">
                    </div>
                    
                    <h3>Additional Plots</h3>
                    <div class="plot-grid">
                        {% if metrics.get("loss") %}
                        <div class="plot-container">
                            <img src="training/plots/loss_comparison.png" alt="Loss Comparison">
                        </div>
                        {% endif %}
                        
                        {% if metrics.get("accuracy") %}
                        <div class="plot-container">
                            <img src="training/plots/accuracy_comparison.png" alt="Accuracy Comparison">
                        </div>
                        {% endif %}
                        
                        {% if metrics.get("lr") %}
                        <div class="plot-container">
                            <img src="training/plots/learning_rate.png" alt="Learning Rate">
                        </div>
                        {% endif %}
                    </div>
                </div>
                {% endif %}
                
                <div class="footer">
                    <p>{{ project_name }} | Deep Learning for Plant Disease Detection</p>
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(template_str)
        return template.render(**context)

    def generate_comparison_report(self, models_data, output_path=None):
        """Generate a comparison report for multiple models

        Args:
            models_data: List of dictionaries with model results
            output_path: Path to save the report
        """
        if output_path is None:
            output_path = self.paths.trials_dir / "model_comparison.html"

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create comparison plots
        plots_dir = Path(os.path.dirname(output_path) / "comparison_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Generate accuracy comparison plot
        accuracy_plot_path = Path(plots_dir / "accuracy_comparison.png")
        self._plot_model_comparison(
            models_data, "test_accuracy", "Test Accuracy", accuracy_plot_path
        )

        # Generate other comparison plots if data is available
        metrics_to_compare = [
            ("test_loss", "Test Loss"),
            ("precision_macro", "Precision (Macro)"),
            ("recall_macro", "Recall (Macro)"),
            ("f1_macro", "F1 Score (Macro)"),
            ("training_time", "Training Time (s)"),
        ]

        plot_paths = {"accuracy": accuracy_plot_path}

        for metric_key, metric_name in metrics_to_compare:
            if any(metric_key in model["metrics"] for model in models_data):
                plot_path = Path(plots_dir / f"{metric_key}_comparison.png")
                self._plot_model_comparison(
                    models_data, metric_key, metric_name, plot_path
                )
                plot_paths[metric_key] = plot_path

        # Create comparison context
        context = {
            "models_data": models_data,
            "plot_paths": plot_paths,
            "generation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "project_name": self.config.get("project", {}).get(
                "name", "Plant Disease Detection"
            ),
            "project_version": self.config.get("project", {}).get("version", "1.0.0"),
        }

        # Generate HTML report
        html = self._render_comparison_template(context)

        # Save report
        with open(output_path, "w") as f:
            f.write(html)

        return output_path

    def _plot_model_comparison(self, models_data, metric_key, metric_name, save_path):
        """Create a bar chart comparing models based on a metric"""
        # Extract data
        model_names = []
        metric_values = []

        for model in models_data:
            model_names.append(model["name"])
            if metric_key in model["metrics"]:
                metric_values.append(model["metrics"][metric_key])
            else:
                metric_values.append(0)

        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, metric_values)

        # Add value annotations
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            if (
                "accuracy" in metric_key
                or "precision" in metric_key
                or "recall" in metric_key
                or "f1" in metric_key
            ):
                text = f"{value:.2%}"
            else:
                text = f"{value:.2f}"
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                text,
                ha="center",
                va="bottom",
                rotation=0,
            )

        plt.title(f"Model Comparison - {metric_name}")
        plt.xlabel("Model")
        plt.ylabel(metric_name)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the plot
        plt.savefig(save_path)
        plt.close()

    def _render_comparison_template(self, context):
        """Render HTML comparison report using Jinja2 template"""
        # Template implementation for comparison report
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Comparison Report</title>
            <style>
                /* CSS styles from the model report template, plus comparison-specific styles */
                /* ... add styles as in the previous template ... */
                .comparison-table {
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 30px;
                }
                .comparison-table th, .comparison-table td {
                    padding: 12px 15px;
                    text-align: center;
                    border: 1px solid #ddd;
                }
                .comparison-table th {
                    background-color: #f8f9fa;
                }
                .comparison-table tr:nth-child(even) {
                    background-color: #f2f2f2;
                }
                .best-value {
                    font-weight: bold;
                    color: #28a745;
                }
                .second-best {
                    color: #17a2b8;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Model Comparison Report</h1>
                    <p>{{ project_name }} v{{ project_version }} | Generated on {{ generation_time }}</p>
                </div>
                
                <div class="card">
                    <h2>Models Compared</h2>
                    <p>Comparison of {{ models_data|length }} models</p>
                    
                    <div class="comparison-table-container">
                        <table class="comparison-table">
                            <thead>
                                <tr>
                                    <th>Model</th>
                                    <th>Test Accuracy</th>
                                    <th>Test Loss</th>
                                    <th>F1 Score</th>
                                    <th>Training Time</th>
                                </tr>
                            </thead>
                            <tbody>
                                {% for model in models_data %}
                                <tr>
                                    <td>{{ model.name }}</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("test_accuracy", 0) * 100) }}%</td>
                                    <td>{{ "%.4f"|format(model.metrics.get("test_loss", 0)) }}</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("f1_macro", 0) * 100) }}%</td>
                                    <td>{{ "%.2f"|format(model.metrics.get("training_time", 0)) }}s</td>
                                </tr>
                                {% endfor %}
                            </tbody>
                        </table>
                    </div>
                </div>
                
                <div class="card">
                    <h2>Visual Comparison</h2>
                    
                    {% if plot_paths.accuracy %}
                    <div class="plot-container">
                        <h3>Accuracy Comparison</h3>
                        <img src="{{ plot_paths.accuracy }}" alt="Accuracy Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.test_loss %}
                    <div class="plot-container">
                        <h3>Loss Comparison</h3>
                        <img src="{{ plot_paths.test_loss }}" alt="Loss Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.f1_macro %}
                    <div class="plot-container">
                        <h3>F1 Score Comparison</h3>
                        <img src="{{ plot_paths.f1_macro }}" alt="F1 Score Comparison">
                    </div>
                    {% endif %}
                    
                    {% if plot_paths.training_time %}
                    <div class="plot-container">
                        <h3>Training Time Comparison</h3>
                        <img src="{{ plot_paths.training_time }}" alt="Training Time Comparison">
                    </div>
                    {% endif %}
                </div>
                
                <div class="summary">
                    <h2>Recommendation</h2>
                    {% set best_model = {'name': '', 'accuracy': 0} %}
                    {% for model in models_data %}
                        {% if model.metrics.get("test_accuracy", 0) > best_model.accuracy %}
                            {% set _ = best_model.update({'name': model.name, 'accuracy': model.metrics.get("test_accuracy", 0)}) %}
                        {% endif %}
                    {% endfor %}
                    
                    <p>Based on the comparison, <strong>{{ best_model.name }}</strong> performs best with an accuracy of {{ "%.2f"|format(best_model.accuracy * 100) }}%.</p>
                </div>
                
                <div class="footer">
                    <p>{{ project_name }} | Deep Learning for Plant Disease Detection</p>
                </div>
            </div>
        </body>
        </html>
        """

        template = Template(template_str)
        return template.render(**context)
```

---

### src/utils/seed_utils.py

```python
# src/utils/seed_utils.py
import os
import random
import numpy as np
import tensorflow as tf


def set_global_seeds(seed=42):
    """Set all seeds for reproducibility

    Args:
        seed: Integer seed value to use

    Returns:
        The seed value used
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Try to make TensorFlow deterministic (may not work on all hardware)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Set session config for older TF versions if needed
    try:
        from tensorflow.keras import backend as K

        if hasattr(tf, "ConfigProto"):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            K.set_session(tf.Session(config=config))
    except:
        pass

    print(f"Global random seeds set to {seed}")
    return seed
```

---

## Summary

Total files: 50
- Python files: 47
- YAML files: 3
