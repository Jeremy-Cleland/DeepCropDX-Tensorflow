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
        
        return weights