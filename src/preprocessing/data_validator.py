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
