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