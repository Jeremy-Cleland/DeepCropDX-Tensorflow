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
    
    return validation_transform