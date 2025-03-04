# import tensorflow as tf
# import os
# import numpy as np
# from pathlib import Path
# from tqdm.auto import tqdm

# from config.config import get_paths
# from src.preprocessing.data_validator import DataValidator


# class DataLoader:
#     def __init__(self, config):
#         """Initialize the data loader with configuration

#         Args:
#             config: Configuration dictionary
#         """
#         self.config = config
#         self.paths = get_paths()

#         # Get training configuration
#         training_config = config.get("training", {})
#         self.batch_size = training_config.get("batch_size", 32)
#         self.validation_split = training_config.get("validation_split", 0.2)
#         self.test_split = training_config.get("test_split", 0.1)

#         # Get hardware configuration
#         hardware_config = config.get("hardware", {})
#         self.num_parallel_calls = hardware_config.get(
#             "num_parallel_calls", tf.data.AUTOTUNE
#         )
#         self.prefetch_size = hardware_config.get(
#             "prefetch_buffer_size", tf.data.AUTOTUNE
#         )

#         # Get data augmentation configuration
#         self.augmentation_config = config.get("data_augmentation", {})

#         # Initialize data validator
#         self.validator = DataValidator(config)

#         # Get validation configuration
#         validation_config = config.get("data_validation", {})
#         self.validate_data = validation_config.get("enabled", True)

#     def load_data(self, data_dir=None):
#         """Load data from the specified directory.
#         Assumes data is organized in a directory structure with each class in its own subdirectory.

#         Args:
#             data_dir: Path to the dataset directory. If None, uses the configured path.

#         Returns:
#             Tuple of (train_dataset, validation_dataset, test_dataset, class_names)
#         """
#         if data_dir is None:
#             # Use configured paths
#             data_path_config = self.config.get("paths", {}).get("data", {})
#             if isinstance(data_path_config, dict):
#                 data_dir = data_path_config.get("processed", "data/processed")
#             else:
#                 data_dir = "data/processed"

#         # Ensure the path is absolute
#         data_dir = Path(data_dir)
#         if not data_dir.is_absolute():
#             data_dir = self.paths.base_dir / data_dir

#         print(f"Loading data from {data_dir}")

#         # Validate the dataset if enabled
#         if self.validate_data:
#             print("Validating dataset before loading...")
#             validation_results = self.validator.validate_dataset(data_dir)

#             # Check for critical errors that would prevent proper training
#             if validation_results["errors"]:
#                 raise ValueError(
#                     f"Dataset validation found critical errors: {validation_results['errors']}. "
#                     "Please fix these issues before training."
#                 )

#             # Log warnings but continue
#             if validation_results["warnings"]:
#                 print("\nDataset validation warnings:")
#                 for warning in validation_results["warnings"]:
#                     print(f"  - {warning}")
#                 print("\nContinuing with data loading despite warnings...\n")

#         # Set up image size based on configuration or default
#         image_size = self.config.get("data", {}).get("image_size", (224, 224))
#         if isinstance(image_size, int):
#             image_size = (image_size, image_size)

#         # Set up data augmentation parameters
#         rotation_range = self.augmentation_config.get("rotation_range", 20)
#         width_shift_range = self.augmentation_config.get("width_shift_range", 0.2)
#         height_shift_range = self.augmentation_config.get("height_shift_range", 0.2)
#         shear_range = self.augmentation_config.get("shear_range", 0.2)
#         zoom_range = self.augmentation_config.get("zoom_range", 0.2)
#         horizontal_flip = self.augmentation_config.get("horizontal_flip", True)
#         vertical_flip = self.augmentation_config.get("vertical_flip", False)
#         fill_mode = self.augmentation_config.get("fill_mode", "nearest")

#         # Set up data generators with augmentation for training
#         train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#             rescale=1.0 / 255,
#             rotation_range=rotation_range,
#             width_shift_range=width_shift_range,
#             height_shift_range=height_shift_range,
#             shear_range=shear_range,
#             zoom_range=zoom_range,
#             horizontal_flip=horizontal_flip,
#             vertical_flip=vertical_flip,
#             fill_mode=fill_mode,
#             validation_split=self.validation_split,
#         )

#         # Load training data with progress bar
#         print("Loading training data...")
#         train_generator = train_datagen.flow_from_directory(
#             data_dir,
#             target_size=image_size,
#             batch_size=self.batch_size,
#             class_mode="categorical",
#             subset="training",
#             shuffle=True,
#         )

#         # Load validation data with progress bar
#         print("Loading validation data...")
#         validation_generator = train_datagen.flow_from_directory(
#             data_dir,
#             target_size=image_size,
#             batch_size=self.batch_size,
#             class_mode="categorical",
#             subset="validation",
#             shuffle=False,
#         )

#         # Create a separate test set if specified
#         test_generator = None
#         if self.test_split > 0:
#             test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
#                 rescale=1.0 / 255
#             )

#             # Check for test directory
#             test_dir = self.paths.data_dir / "test"
#             if test_dir.exists():
#                 print("Loading test data from dedicated test directory...")
#                 test_generator = test_datagen.flow_from_directory(
#                     test_dir,
#                     target_size=image_size,
#                     batch_size=self.batch_size,
#                     class_mode="categorical",
#                     shuffle=False,
#                 )
#             else:
#                 # Split validation data into validation and test
#                 print("Dedicated test directory not found.")
#                 print(
#                     f"Using {self.test_split/(1-self.validation_split):.1%} of validation data as test set..."
#                 )

#                 # This is a simplified approach. In a real application, you would
#                 # want to create a proper split of the data.
#                 test_generator = validation_generator
#         else:
#             print("Test split not configured, skipping test data loading")

#         # Get class names
#         class_indices = train_generator.class_indices
#         class_names = {v: k for k, v in class_indices.items()}

#         # Print dataset statistics
#         print(f"Dataset loaded successfully:")
#         print(f"  - Training: {train_generator.samples} images")
#         print(f"  - Validation: {validation_generator.samples} images")
#         if test_generator:
#             print(f"  - Test: {test_generator.samples} images")
#         print(f"  - Classes: {len(class_names)} ({', '.join(class_names.values())})")
#         print(f"  - Image size: {image_size}")
#         print(f"  - Batch size: {self.batch_size}")

#         return train_generator, validation_generator, test_generator, class_names

#     def preprocess_function(self, image, label):
#         """Apply preprocessing to a single example

#         Args:
#             image: Image tensor
#             label: Label tensor

#         Returns:
#             Tuple of (processed_image, label)
#         """
#         # Get preprocessing configuration
#         preprocessing_config = self.config.get("preprocessing", {})

#         # Normalize if not already done
#         if preprocessing_config.get("normalize", True):
#             image = tf.cast(image, tf.float32) / 255.0

#         # Apply additional preprocessing steps as configured
#         if preprocessing_config.get("center_crop", False):
#             # Center crop to target size
#             target_size = preprocessing_config.get("target_size", (224, 224))
#             image = tf.image.resize_with_crop_or_pad(
#                 image, target_size[0], target_size[1]
#             )

#         if preprocessing_config.get("standardize", False):
#             # Standardize to mean 0, std 1
#             image = tf.image.per_image_standardization(image)

#         return image, label

#     def apply_data_pipeline(self, dataset):
#         """Apply preprocessing to a dataset with progress tracking

#         Args:
#             dataset: TensorFlow dataset

#         Returns:
#             Processed dataset
#         """
#         # Count items for progress bar
#         total_items = sum(1 for _ in dataset.take(-1))

#         with tqdm(total=total_items, desc="Preprocessing") as pbar:
#             # Create a dataset that updates the progress bar
#             def update_progress(*args):
#                 pbar.update(1)
#                 return args

#             # Apply preprocessing with progress updates
#             dataset = dataset.map(
#                 self.preprocess_function, num_parallel_calls=self.num_parallel_calls
#             )
#             dataset = dataset.map(update_progress)

#             # Apply batching and prefetching
#             dataset = dataset.batch(self.batch_size)
#             dataset = dataset.prefetch(self.prefetch_size)

#         return dataset

# continued from src/preprocessing/data_loader_tf.py
# src/preprocessing/data_loader_tf.py

import tensorflow as tf
import os
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import glob

from src.config.config import get_paths
from src.preprocessing.data_validator import DataValidator
from src.utils.seed_utils import set_global_seeds


class TFDataLoader:
    """
    Enhanced data loader using tf.data API for better performance and memory efficiency.
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

        # Initialize data validator
        self.validator = DataValidator(config)

        # Get validation configuration
        validation_config = config.get("data_validation", {})
        self.validate_data = validation_config.get("enabled", True)

        # Set image parameters
        self.image_size = self.config.get("data", {}).get("image_size", (224, 224))
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)

    def load_data_efficient(self, data_dir=None):
        """Load data using efficient tf.data pipeline with sharding support

        Args:
            data_dir: Path to the dataset directory. If None, uses the configured path.

        Returns:
            Tuple of (train_dataset, validation_dataset, test_dataset, class_names)
        """
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

        # Batch and prefetch datasets
        train_dataset = train_dataset.batch(self.batch_size).prefetch(
            self.prefetch_size
        )
        val_dataset = val_dataset.batch(self.batch_size).prefetch(self.prefetch_size)
        if test_dataset is not None:
            test_dataset = test_dataset.batch(self.batch_size).prefetch(
                self.prefetch_size
            )

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
        return train_dataset, val_dataset, test_dataset, class_names

    # Keep the original method for backward compatibility
    def load_data(self, data_dir=None):
        """
        Legacy method that calls the more efficient version.
        Kept for backwards compatibility.
        """
        return self.load_data_efficient(data_dir)
