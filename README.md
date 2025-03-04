# Plant Disease Detection Project - Restructured

## Directory Structure

```
root/
├── src/                        # All source code
│   ├── config/                 # Configuration management
│   │   ├── config.py           # New path management system
│   │   ├── config_loader.py    # Load YAML configs
│   │   ├── model_configs/      # Model-specific configurations
│   │   │   └── models.yaml     # Centralized model configs
│   ├── evaluation/             # Evaluation utilities
│   │   ├── metrics.py          # Metrics calculation
│   │   └── visualization.py    # Visualization utilities
│   ├── models/                 # Model definitions
│   │   └── model_factory.py    # Model creation factory
│   ├── preprocessing/          # Data preprocessing
│   │   └── data_loader.py      # Data loading and preprocessing
│   ├── scripts/                # Executable scripts
│   │   ├── train.py            # Training script
│   │   ├── evaluate.py         # Evaluation script
│   │   └── compare_models.py   # Model comparison script
│   ├── training/               # Training utilities
│   │   └── trainer.py          # Model trainer with tqdm
│   ├── utils/                  # Utility functions
│   │   ├── logger.py           # Logging utilities
│   │   └── report_generator.py # Report generation
│   └── main.py                 # Main entry point
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
├── logs/                       # General logs
└── restructure_project.py      # Script to restructure the project
```

## Key Improvements

1. **Centralized Path Management**:
   - Created `config.py` with `ProjectPaths` class to manage all project paths
   - All modules use a singleton path instance for consistency
   - Directories are automatically created when needed

2. **Organized Trial Results**:
   - Each model's training runs are stored in separate directories
   - Organized structure for training and evaluation artifacts
   - Timestamped run IDs to track experiments

3. **Progress Tracking**:
   - Added tqdm progress bars for training, evaluation, and data loading
   - Better visibility into long-running operations

4. **Enhanced Main Script**:
   - Support for running one model, multiple models, or all models
   - Command-line arguments for flexibility
   - Improved error handling and reporting

5. **Evaluation Enhancements**:
   - Comprehensive metrics calculation
   - Rich visualizations including confusion matrices, ROC curves, etc.
   - HTML reports for easier interpretation

## Using the New Structure

1. **Running the main script**:

   ```bash
   # Train a single model
   python -m src.main --model ResNet50 --data_dir data/processed

   # Train multiple specific models
   python -m src.main --models EfficientNetB0 EfficientNetB1 MobileNetV2 --data_dir data/processed

   # Train all models defined in the configuration
   python -m src.main --all_models --data_dir data/processed
   ```

2. **Running individual scripts**:

   ```bash
   # Train a specific model
   python -m src.scripts.train --model ResNet50 --data_dir data/processed

   # Evaluate a trained model
   python -m src.scripts.evaluate --model_path trials/ResNet50/run_20250304_123456_001/ResNet50_final.h5 --data_dir data/test

   # Compare multiple trained models
   python -m src.scripts.compare_models --models ResNet50 EfficientNetB0 MobileNetV2
   ```

3. **Restructuring Existing Project**:

   ```bash
   # Dry run to see what would happen
   python restructure_project.py --dry_run

   # Actual restructuring
   python restructure_project.py
   ```

## Next Steps

1. Update any imports in the files to reference the new structure
2. Add the `src` directory to your Python path or install as a package
3. Run a test training to ensure everything works correctly
4. Consider creating a requirements.txt or updating your environment.yml file

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
