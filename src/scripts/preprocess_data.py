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
        if src_path.lower().endswith((".jpg", ".jpeg")):
            img = tf.image.decode_jpeg(img, channels=3)
        elif src_path.lower().endswith(".png"):
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


def preprocess_dataset(
    raw_dir, processed_dir, target_size=(224, 224), num_workers=4, copy_only=False
):
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
        image_files = (
            list(raw_dir.glob("*.jpg"))
            + list(raw_dir.glob("*.jpeg"))
            + list(raw_dir.glob("*.png"))
        )
        if image_files:
            print(
                f"Found {len(image_files)} images directly in {raw_dir}. These should be organized into class directories."
            )
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
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(tasks),
                desc=f"Processing {class_name}",
            ):
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
    parser.add_argument(
        "--raw_dir", type=str, default="data/raw", help="Directory with raw images"
    )
    parser.add_argument(
        "--processed_dir",
        type=str,
        default="data/processed",
        help="Directory for processed images",
    )
    parser.add_argument(
        "--height", type=int, default=224, help="Target height for resizing"
    )
    parser.add_argument(
        "--width", type=int, default=224, help="Target width for resizing"
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of worker threads"
    )
    parser.add_argument(
        "--copy_only", action="store_true", help="Just copy files without processing"
    )

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
        copy_only=args.copy_only,
    )


if __name__ == "__main__":
    main()
