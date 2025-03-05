
![Crop Disease Detection Banner](.github/assets/crop_disease_detection_01.jpg)

# **DeepCropDX | Advanced Plant Disease Detection System**

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DeepCropDX is a comprehensive deep learning pipeline for accurately diagnosing plant diseases from images. The system combines state-of-the-art neural network architectures (DenseNet, EfficientNet, MobileNet, ResNet, and Xception) with transfer learning to deliver highly accurate disease detection for agricultural applications.

## Directory Structure

```
root/
├── src/                        # All source code
│   ├── config/                 # Configuration management
│   │   ├── config.py           # Path management system
│   │   ├── config_loader.py    # Load YAML configs
│   │   ├── config_manager.py   # Manage config and CLI args
│   │   ├── model_configs/      # Model-specific configurations
│   │   │   └── models.yaml     # Centralized model configs
│   ├── evaluation/             # Evaluation utilities
│   │   ├── metrics.py          # Metrics calculation
│   │   └── visualization.py    # Visualization utilities
│   ├── models/                 # Model definitions
│   │   ├── model_factory.py    # Basic model creation factory
│   │   ├── model_factory_new.py # Enhanced model factory
│   │   ├── model_optimizer.py  # Model quantization and pruning
│   │   └── advanced_architectures.py # Modern model architectures
│   ├── preprocessing/          # Data preprocessing
│   │   ├── data_loader.py      # Basic data loading
│   │   ├── data_loader_new.py  # Enhanced data loading
│   │   ├── data_transformations.py # Advanced data augmentations
│   │   └── dataset_pipeline.py # Optimized data pipeline
│   ├── scripts/                # Executable scripts
│   │   ├── train.py            # Training script
│   │   ├── evaluate.py         # Evaluation script
│   │   └── compare_models.py   # Model comparison script
│   ├── training/               # Training utilities
│   │   ├── trainer.py          # Model trainer
│   │   ├── batch_trainer.py    # Batch training capabilities
│   │   ├── training_pipeline.py # Training orchestration
│   │   └── learning_rate_scheduler.py # Advanced LR scheduling
│   ├── utils/                  # Utility functions
│   │   ├── logger.py           # Logging utilities
│   │   ├── report_generator.py # Report generation
│   │   ├── cli_utils.py        # CLI argument handling
│   │   ├── error_handling.py   # Error handling utilities
│   │   └── memory_utils.py     # Memory management utilities
│   └── main.py                 # Main entry point (simplified)
├── data/                       # Data storage
│   ├── raw/                    # Raw data
│   ├── processed/              # Processed data
│   └── test/                   # Test data (optional)
├── trials/                     # Model training results
│   ├── ModelName1/             # Results for a specific model
│   │   └── run_YYYYMMDD_HHMMSS_001/  # Specific training run
│   │       ├── config.json     # Configuration used
│   │       ├── final_metrics.json  # Final performance metrics
│   │       ├── model_summary.txt  # Model architecture summary
│   │       ├── report.html     # HTML report
│   │       ├── ModelName_final.h5  # Saved model
│   │       ├── training/       # Training artifacts
│   │       │   ├── checkpoints/  # Model checkpoints
│   │       │   ├── plots/      # Training plots
│   │       │   └── tensorboard/  # TensorBoard logs
│   │       └── evaluation/     # Evaluation artifacts
│   │           ├── metrics.json  # Evaluation metrics
│   │           └── plots/      # Evaluation plots
│   └── ModelName2/             # Another model's results
├── docs/                       # Documentation
│   └── optimization.md         # Documentation for optimization features
├── config/                     # Configuration examples
│   └── examples/               
│       └── lr_schedule_config.yaml # Example LR schedule config
├── logs/                       # General logs
└── restructure_project.py      # Script to restructure the project
```

## Key Improvements

1. **Modular Architecture**:
   - Simplified main.py by separating it into focused modules
   - Clear separation of concerns between CLI, training, and reporting
   - Improved maintainability and testability

2. **Enhanced Error Handling**:
   - Comprehensive error handling throughout the codebase
   - Custom exception types for specific error scenarios
   - Error recovery with retry capabilities
   - Detailed error reporting and logging

3. **Memory Management**:
   - Advanced memory cleanup between model training runs
   - Memory usage monitoring and reporting
   - GPU memory optimization for efficient batch training
   - Prevention of memory leaks during long training sessions

4. **Advanced Model Features**:
   - Support for model quantization (post-training and during-training)
   - Model pruning capabilities for smaller deployments
   - Learning rate warmup scheduling for better convergence
   - Support for newer model architectures (EfficientNetV2, ConvNeXt, ViT)

5. **Optimized Data Pipeline**:
   - Refactored data loading to separate concerns
   - Enhanced performance with prefetching and parallel processing
   - Extensive data augmentation options
   - Improved memory efficiency during data processing

6. **Comprehensive Documentation**:
   - Enhanced type hints and detailed docstrings
   - Documentation for optimization techniques
   - Example configuration files
   - Improved code organization

## Using the New Structure

1. **Running the main script**:

   ```bash
   # Train a single model
   python -m src.main --model ResNet50 --data_dir data/processed

   # Train multiple specific models
   python -m src.main --models EfficientNetB0 EfficientNetB1 MobileNetV2 --data_dir data/processed

   # Train all models defined in the configuration
   python -m src.main --all_models --data_dir data/processed
   
   # Use advanced learning rate scheduling
   python -m src.main --model ResNet50 --config config/examples/lr_schedule_config.yaml
   
   # Print hardware information and exit
   python -m src.main --hardware_info
   ```

2. **Using optimization features**:

   ```bash
   # Enable model quantization
   python -m src.main --model MobileNetV2 --config config/examples/quantization_config.yaml

   # Enable model pruning
   python -m src.main --model ResNet50 --config config/examples/pruning_config.yaml
   
   # Combined optimizations
   python -m src.main --model EfficientNetB0 --config config/examples/optimized_training.yaml
   ```

3. **Advanced data pipeline usage**:

   ```bash
   # Use enhanced data loading with memory optimization
   python -m src.main --model ResNet50 --use_new_data_loader --data_dir data/processed
   
   # Enable advanced data augmentations
   python -m src.main --model MobileNetV2 --advanced_augmentations --data_dir data/processed
   ```

4. **Memory management options**:

   ```bash
   # Enable explicit GPU memory management
   python -m src.main --model ResNet50 --manage_gpu_memory
   
   # Monitor memory usage during training
   python -m src.main --model ResNet50 --monitor_memory
   ```

## Next Steps

1. Update any imports in the files to reference the new structure
2. Add the `src` directory to your Python path or install as a package
3. Run a test training to ensure everything works correctly
4. Consider creating a requirements.txt or updating your environment.yml file

# New Optimization Features

## Overview

The project now includes several advanced optimization features to improve model performance, reduce memory usage, and enhance training efficiency.

## Model Quantization

Model quantization reduces model size and improves inference speed by representing weights with lower precision.

```python
from src.models.model_optimizer import ModelOptimizer

# Create optimizer
optimizer = ModelOptimizer(config)

# Apply post-training quantization
quantized_model = optimizer.apply_quantization(
    model, 
    representative_dataset=representative_dataset,
    method="post_training",
    quantization_bits=8
)
```

## Model Pruning

Pruning removes redundant weights to create sparse models that are smaller and potentially faster.

```python
from src.models.model_optimizer import ModelOptimizer

# Create optimizer
optimizer = ModelOptimizer(config)

# Apply pruning during training
pruning_params = optimizer.setup_pruning(
    model,
    target_sparsity=0.5,
    pruning_schedule="polynomial"
)

# Get pruning callbacks for training
pruning_callbacks = optimizer.get_pruning_callbacks()
```

## Learning Rate Scheduling

Advanced learning rate scheduling for improved training stability and convergence.

```python
from src.training.learning_rate_scheduler import get_warmup_scheduler

# Get warmup scheduler
scheduler = get_warmup_scheduler(config)

# Add to training callbacks
callbacks.append(scheduler)
```

## Memory Management

Utilities for monitoring and optimizing memory usage during training.

```python
from src.utils.memory_utils import clean_memory, log_memory_usage

# Log current memory usage
memory_stats = log_memory_usage(prefix="Before training: ")

# Clean up memory after training
clean_memory(clean_gpu=True)
```

# Model Registry System

## Overview

The Model Registry is a centralized system for tracking, comparing, and managing all your trained models. It automatically captures metadata about each training run and provides tools to easily find and load the best performing models.

## Key Features

### 1. Automatic Model Tracking

- Seamlessly integrated with the `Trainer` class to automatically register models
- Captures key metadata: accuracy, loss, training time, model architecture
- Tracks the location of model files, checkpoints, and TensorBoard logs

### 2. Model Management

- Maintains a registry of all trained models and their versions
- Identifies best-performing models based on configurable metrics
- Supports model versioning with timestamped run IDs

### 3. Performance Comparison

- Compare different models side-by-side
- Generate comparative visualizations of key metrics
- Easily identify which models and hyperparameters work best

### 4. Command-Line Interface

- `registry_cli.py` provides a full-featured CLI for registry management
- List models, view details, generate reports, and more
- Perform model comparisons from the command line

### 5. Reporting

- Generate HTML reports of your model inventory
- View detailed metrics and performance statistics
- Keep track of your experiments and results

## Directory Structure

The registry maintains a structured organization for all model artifacts:

```
trials/
├── registry.json              # Central registry database
├── registry_report.html       # Generated HTML report
├── comparisons/               # Model comparison visualizations
│   ├── comparison_test_accuracy.png
│   └── comparison_combined.png
├── ModelName1/                # Model-specific directory
│   └── run_YYYYMMDD_HHMMSS_001/  # Run-specific directory
│       ├── ModelName1_final.h5   # Saved model file
│       ├── final_metrics.json    # Performance metrics
│       ├── model_summary.txt     # Architecture summary
│       ├── training/             # Training artifacts
│       │   ├── checkpoints/      # Model checkpoints
│       │   ├── plots/            # Training plots
│       │   └── tensorboard/      # TensorBoard logs
│       └── evaluation/           # Evaluation artifacts
│           ├── metrics.json      # Evaluation metrics
│           └── plots/            # Evaluation plots
└── ModelName2/                # Another model
```

## Using the Model Registry

### Programmatic Usage

```python
from model_registry.registry_manager import ModelRegistryManager

# Initialize the registry
registry = ModelRegistryManager()

# Scan for models that aren't in the registry yet
registry.scan_trials()

# Get the best performing model
model = registry.get_model("ResNet50", best=True)

# Get information about a model's runs
run_info = registry.get_run_info("EfficientNetB0")

# Compare models
comparison = registry.compare_models(
    model_names=["ResNet50", "EfficientNetB0", "MobileNetV2"],
    metrics=["test_accuracy", "training_time"]
)
```

### Command-Line Usage

```bash
# List all models in the registry
python -m src.scripts.registry_cli list

# Show details for a specific model
python -m src.scripts.registry_cli details --model ResNet50

# Compare multiple models
python -m src.scripts.registry_cli compare --models ResNet50 EfficientNetB0 MobileNetV2

# Generate an HTML report of all models
python -m src.scripts.registry_cli report

# Scan for new models and runs
python -m src.scripts.registry_cli scan
```

## Integration with Training

The Model Registry is automatically integrated with the training process. When you train a model using the `Trainer` class, it will:

1. Register the model in the registry
2. Track key performance metrics
3. Save all necessary metadata
4. Update registry records

No additional code is needed - it just works!

## Benefits

- **Reproducibility**: Easily find and reproduce your best models
- **Organization**: Keep track of all your experiments in one place
- **Efficiency**: Quickly identify which approaches work best
- **Collaboration**: Share results with team members through standardized reports
