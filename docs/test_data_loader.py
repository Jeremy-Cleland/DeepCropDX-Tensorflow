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
print("You can now use data_loader.py in your main code once TensorFlow is installed.")