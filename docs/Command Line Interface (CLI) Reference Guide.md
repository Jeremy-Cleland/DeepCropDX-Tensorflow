# DeepCropDX: Command Line Interface (CLI) Reference Guide

## Table of Contents

- [DeepCropDX: Command Line Interface (CLI) Reference Guide](#deepcropdx-command-line-interface-cli-reference-guide)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Training Commands](#training-commands)
    - [Basic Training](#basic-training)
    - [Learning Rate Configuration](#learning-rate-configuration)
    - [Model Attention Mechanisms](#model-attention-mechanisms)
    - [Model Optimization](#model-optimization)
  - [Preprocessing Commands](#preprocessing-commands)
    - [Data Preparation](#data-preparation)
    - [Processing Options](#processing-options)
  - [Evaluation Commands](#evaluation-commands)
    - [Model Assessment](#model-assessment)
    - [Visualization Options](#visualization-options)
  - [Model Registry Commands](#model-registry-commands)
    - [Model Management](#model-management)
    - [Comparative Analysis](#comparative-analysis)
    - [Registry Maintenance](#registry-maintenance)
  - [Project Maintenance Commands](#project-maintenance-commands)
    - [Cleanup Operations](#cleanup-operations)
    - [Safety Options](#safety-options)
  - [Configuration System](#configuration-system)
    - [Global Configuration](#global-configuration)
    - [Main Configuration Options](#main-configuration-options)
      - [Project Information](#project-information)
      - [Training Parameters](#training-parameters)
      - [Early Stopping](#early-stopping)
      - [Learning Rate Scheduling](#learning-rate-scheduling)
      - [Hardware Optimization](#hardware-optimization)
      - [Logging \& Reporting](#logging--reporting)
      - [Data Configuration](#data-configuration)
      - [Data Augmentation](#data-augmentation)
      - [Data Validation](#data-validation)
      - [Model Optimization](#model-optimization-1)
        - [Quantization](#quantization)
        - [Pruning](#pruning)
    - [Model-Specific Configuration](#model-specific-configuration)
      - [Base Model Configuration](#base-model-configuration)
      - [Fine-Tuning Configuration](#fine-tuning-configuration)
      - [Preprocessing Configuration](#preprocessing-configuration)
      - [Model-Specific Hyperparameters](#model-specific-hyperparameters)
  - [Performance Considerations](#performance-considerations)
    - [Quantization Performance](#quantization-performance)
    - [Pruning Performance](#pruning-performance)
    - [Learning Rate Scheduling Performance](#learning-rate-scheduling-performance)
    - [Attention Mechanisms Performance](#attention-mechanisms-performance)
    - [Hardware Support](#hardware-support)
    - [Combined Optimization Strategy](#combined-optimization-strategy)
  - [Best Practices \& Recommendations](#best-practices--recommendations)

## Introduction

DeepCropDX is a comprehensive deep learning framework for plant disease detection, implementing state-of-the-art image processing techniques and neural network architectures. This reference guide documents all command-line interfaces available in the project, providing detailed information on parameters, usage patterns, and performance considerations.

The command-line tools in DeepCropDX provide functionality for:

- Training deep learning models with various architectures and optimizations
- Preprocessing image data for model training
- Evaluating model performance with detailed metrics
- Managing a model registry for tracking and comparing multiple model runs
- Project maintenance and resource management

Each command described in this document includes details on required and optional parameters, practical examples, and performance considerations based on empirical evaluations.

## Training Commands

The training commands provide functionality for creating, training, and optimizing deep learning models for plant disease detection.

### Basic Training

`train.py` is the primary script for training models. It provides options for selecting model architecture, configuring training parameters, and managing the training process.

**Usage:**

```bash
python -m src.scripts.train --model MODEL_NAME [options]
```

**Required Parameters:**

- `--model MODEL_NAME`: Model architecture to train (e.g., ResNet50, EfficientNetB0)

**Optional Parameters:**

- `--data_dir PATH`: Path to dataset directory (default: config-specified path)
- `--config PATH`: Path to custom configuration file
- `--batch_size SIZE`: Override batch size for training
- `--epochs NUM`: Override number of epochs for training
- `--learning_rate RATE`: Override learning rate
- `--seed NUM`: Random seed for reproducibility (default: 42)
- `--resume`: Resume training from latest checkpoint
- `--hardware_summary`: Print hardware configuration and exit

**Examples:**

```bash
# Train a ResNet50 model with default parameters
python -m src.scripts.train --model ResNet50

# Train with custom dataset location and hyperparameters
python -m src.scripts.train --model EfficientNetB0 --data_dir data/my_dataset --batch_size 16 --epochs 100

# Resume training from checkpoint
python -m src.scripts.train --model MobileNetV2 --resume
```

### Learning Rate Configuration

The training script provides several options for learning rate configuration, which can significantly impact model performance.

**Learning Rate Finder:**

```bash
python -m src.scripts.train --model ResNet50 --find_lr
```

This runs the learning rate finder before training to automatically determine an optimal learning rate value based on the loss curve analysis.

**Learning Rate Scheduling:**

```bash
python -m src.scripts.train --model EfficientNetB0 --warmup_epochs 5
```

Enables learning rate warmup for the specified number of epochs, followed by a decay schedule.

### Model Attention Mechanisms

DeepCropDX supports adding attention mechanisms to standard architectures, which can improve model performance by focusing on relevant image regions.

**Available Attention Types:**

- `se`: Squeeze-and-Excitation blocks (channel attention)
- `cbam`: Convolutional Block Attention Module (channel + spatial attention)
- `spatial`: Spatial attention only

**Usage:**

```bash
python -m src.scripts.train --model ResNet50 --attention se
```

**Pre-configured Enhanced Models:**

```bash
python -m src.scripts.train --model ResNet50_CBAM --use_enhanced
```

Uses a pre-configured model with attention mechanisms built-in.

### Model Optimization

DeepCropDX supports model optimization techniques like quantization and pruning to create smaller, faster models.

**Quantization:**

```bash
python -m src.scripts.train --model MobileNetV2 --quantize
```

Enables post-training quantization to reduce model size and improve inference speed.

**Pruning:**

```bash
python -m src.scripts.train --model EfficientNetB0 --pruning
```

Enables model pruning during training to create sparse models.

**Combined Optimizations:**

```bash
python -m src.scripts.train --model MobileNetV2 --quantize --pruning --warmup_epochs 5
```

Combines multiple optimization techniques for maximum efficiency.

## Preprocessing Commands

The preprocessing commands allow you to prepare image data for model training, including resizing, conversion, and organization.

### Data Preparation

`preprocess_data.py` is the main script for preprocessing raw image data into the format expected by the training pipeline.

**Usage:**

```bash
python -m src.scripts.preprocess_data [options]
```

**Optional Parameters:**

- `--raw_dir PATH`: Directory with raw images (default: data/raw)
- `--processed_dir PATH`: Directory for processed images (default: data/processed)
- `--height SIZE`: Target height for resizing (default: 224)
- `--width SIZE`: Target width for resizing (default: 224)
- `--workers NUM`: Number of worker threads (default: 4)
- `--copy_only`: Just copy files without processing

**Examples:**

```bash
# Preprocess with default settings
python -m src.scripts.preprocess_data

# Preprocess with custom dimensions and paths
python -m src.scripts.preprocess_data --raw_dir /path/to/raw_dataset --processed_dir /path/to/output --height 299 --width 299

# Use more worker threads for faster processing
python -m src.scripts.preprocess_data --workers 8

# Copy files without resizing (for pre-processed datasets)
python -m src.scripts.preprocess_data --copy_only
```

### Processing Options

The preprocessing script supports various options to control how images are processed:

**Multithreaded Processing:**  
The `--workers` parameter controls the number of parallel processing threads. Higher values can significantly speed up preprocessing on multi-core systems, but may increase memory usage.

**Image Resizing:**  
All images will be resized to the specified dimensions. This is important for maintaining consistent input sizes for the models.

**Copy-Only Mode:**  
When using `--copy_only`, images are copied directly without processing. This is useful when:

- Your images are already properly sized and formatted
- You want to preserve the exact original image data
- You're reorganizing an existing dataset

## Evaluation Commands

The evaluation commands provide functionality to assess model performance, generate metrics, and create visualizations for analysis.

### Model Assessment

`evaluate.py` is the primary script for evaluating trained models against test datasets.

**Usage:**

```bash
python -m src.scripts.evaluate --model_path PATH_TO_MODEL [options]
```

**Required Parameters:**

- `--model_path PATH`: Path to the trained model (.h5 file)

**Optional Parameters:**

- `--data_dir PATH`: Path to the dataset directory for evaluation
- `--config PATH`: Path to the configuration file
- `--output_dir PATH`: Directory to save evaluation results
- `--batch_size SIZE`: Batch size for evaluation (overrides config)
- `--visualize_misclassified`: Generate visualizations of misclassified samples

**Examples:**

```bash
# Basic evaluation
python -m src.scripts.evaluate --model_path trials/ResNet50/run_20250304_123456/ResNet50_final.h5

# Evaluate with custom dataset and output directory
python -m src.scripts.evaluate --model_path trials/EfficientNetB0/run_latest/EfficientNetB0_final.h5 --data_dir data/validation_set --output_dir results/evaluation

# With visualization of misclassified examples
python -m src.scripts.evaluate --model_path models/best_model.h5 --visualize_misclassified
```

### Visualization Options

The evaluation script generates several visualizations to help understand model performance:

**Standard Visualizations:**

- Confusion matrix (both raw and normalized)
- ROC curves for all classes
- Precision-recall curves
- Class distribution histogram

**Misclassified Examples:**  
When using the `--visualize_misclassified` option, the script will generate a grid of misclassified images with their true and predicted labels, helping to identify patterns in model errors.

**HTML Report:**  
The evaluation script automatically generates a comprehensive HTML report containing all metrics and visualizations, making it easy to share and review results.

## Model Registry Commands

The model registry commands allow you to manage, compare, and analyze multiple model runs, providing a centralized view of all training experiments.

### Model Management

`registry_cli.py` provides a command-line interface for interacting with the model registry.

**List Models:**

```bash
python -m src.scripts.registry_cli list
```

Lists all models in the registry along with their key metrics.

**List Runs:**

```bash
python -m src.scripts.registry_cli runs --model ResNet50
```

Lists all runs for a specific model.

**Show Details:**

```bash
python -m src.scripts.registry_cli details --model ResNet50 --run run_20250304_123456_001
```

Shows detailed information about a specific model run.

**Scan for New Models:**

```bash
python -m src.scripts.registry_cli scan
```

Scans the trials directory for new models and runs to add to the registry.

### Comparative Analysis

The registry provides tools for comparing model performance across different architectures and configurations.

**Compare Models:**

```bash
python -m src.scripts.registry_cli compare --models ResNet50 MobileNetV2 EfficientNetB0
```

Compares specified models based on key metrics.

**Compare Top Models:**

```bash
python -m src.scripts.registry_cli compare --top 5
```

Compares the top N models based on performance.

**Custom Metrics Comparison:**

```bash
python -m src.scripts.registry_cli compare --models ResNet50 MobileNetV2 --metrics test_accuracy precision_macro recall_macro
```

Compares models using specific metrics.

### Registry Maintenance

The registry CLI provides commands for maintaining and managing the registry database.

**Export Registry:**

```bash
python -m src.scripts.registry_cli export --output registry_backup.json
```

Exports the registry to a file for backup.

**Import Registry:**

```bash
python -m src.scripts.registry_cli import --input registry_backup.json
```

Imports a registry from a file.

**Generate Report:**

```bash
python -m src.scripts.registry_cli report --output registry_report.html
```

Generates an HTML report summarizing the registry contents.

**Delete Run:**

```bash
python -m src.scripts.registry_cli delete --model ResNet50 --run run_20250304_123456_001
```

Deletes a run from the registry.

**Delete Run and Files:**

```bash
python -m src.scripts.registry_cli delete --model ResNet50 --run run_20250304_123456_001 --delete-files --force
```

Deletes a run from the registry and removes associated files from disk.

## Project Maintenance Commands

The project maintenance commands help manage resources, clear cached files, and ensure a clean state for experiments.

### Cleanup Operations

`cleanup.py` is the main script for cleaning up generated files and artifacts.

**Usage:**

```bash
python -m cleanup [options]
```

**Optional Parameters:**

- `--all`: Remove all generated files (models, reports, logs, etc.)
- `--models`: Remove only model files and directories
- `--reports`: Remove only report files and directories
- `--logs`: Remove only log files
- `--pycache`: Remove only **pycache** directories
- `--keep-registry`: Keep the model registry file when cleaning models

**Examples:**

```bash
# Clean everything
python -m cleanup --all

# Clean only model files but keep the registry
python -m cleanup --models --keep-registry

# Clean __pycache__ directories only
python -m cleanup --pycache
```

### Safety Options

The cleanup script includes safety features to prevent accidental deletion of important files.

**Dry Run:**

```bash
python -m cleanup --all --dry-run
```

Shows what would be removed without actually deleting files.

**Confirmation Prompt:**
By default, the script will prompt for confirmation before deleting files. This can be bypassed with:

```bash
python -m cleanup --all --confirm
```

Skips the confirmation prompt (use with caution).

**Safe Path Checks:**
The script includes safety checks to ensure that:

- Source code directories are not deleted (except for **pycache**)
- Raw data directories are preserved
- Only files within the project root are affected

## Configuration System

DeepCropDX uses a hierarchical configuration system based on YAML files. This section details the key configuration options that control model training and evaluation.

### Global Configuration

Most command-line tools accept a `--config` parameter to specify a custom configuration file:

```bash
python -m src.scripts.train --model ResNet50 --config configs/custom_config.yaml
```

### Main Configuration Options

The main configuration file (`config.yaml`) controls project-wide settings and defaults. Key sections include:

#### Project Information

```yaml
project:
  name: Plant Disease Detection
  description: Deep learning models for detecting diseases in plants
  version: 1.0.0
```

#### Training Parameters

```yaml
training:
  batch_size: 128
  epochs: 50
  learning_rate: 0.001
  optimizer: adam         # Options: adam, sgd, rmsprop, adagrad, adadelta
  loss: categorical_crossentropy
  metrics: [accuracy, AUC, Precision, Recall]
  class_weight: "balanced" # Handles class imbalance automatically
  clip_norm: 1.0          # Gradient clipping by norm
  clip_value: null        # Gradient clipping by value
```

#### Early Stopping

```yaml
early_stopping:
  enabled: true
  patience: 10            # Epochs to wait before stopping
  monitor: val_loss       # Options: val_loss, val_accuracy, etc.
  validation_split: 0.2   # Fraction for validation
  test_split: 0.1         # Fraction for test set
  progress_bar: true      # Enable tqdm progress bar
```

#### Learning Rate Scheduling

```yaml
lr_schedule:
  enabled: false
  type: "warmup_cosine"   # Options: warmup_cosine, warmup_exponential, warmup_step, one_cycle
  warmup_epochs: 5
  min_lr: 1.0e-6
  factor: 0.5             # Decay factor
  patience: 5             # For ReduceLROnPlateau
```

#### Hardware Optimization

```yaml
hardware:
  use_metal: true         # For Apple Silicon GPUs
  mixed_precision: true   # Use FP16/FP32 mixed precision training
  memory_growth: true     # Prevent TensorFlow from allocating all GPU memory
  num_parallel_calls: 16  # Parallel threads for data loading
  prefetch_buffer_size: 8 # Prefetched batches in pipeline
```

#### Logging & Reporting

```yaml
logging:
  level: INFO             # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
  tensorboard: true       # Enable TensorBoard logging
  separate_loggers: true  # Use separate logs for training and evaluation
  
reporting:
  generate_plots: true             # Generate training history plots
  save_confusion_matrix: true      # Save confusion matrix visualization
  save_roc_curves: true            # Save ROC curves
  save_precision_recall: true      # Save precision-recall curves
  generate_html_report: true       # Generate comprehensive HTML report
```

#### Data Configuration

```yaml
data:
  image_size: [224, 224]  # Input image dimensions
  save_splits: true       # Save dataset splits to disk
  use_saved_splits: true  # Use saved splits when available
  splits_dir: "splits"    # Directory for storing splits
  cache_parsed_images: false # Cache decoded images in memory
```

#### Data Augmentation

```yaml
data_augmentation:
  enabled: true
  rotation_range: 20        # Max rotation angle (degrees)
  width_shift_range: 0.2    # Max horizontal shift (fraction of width)
  height_shift_range: 0.2   # Max vertical shift (fraction of height)
  shear_range: 0.2          # Shear intensity (radians)
  zoom_range: 0.2           # Random zoom range
  horizontal_flip: true     # Enable horizontal flipping
  vertical_flip: false      # Disable vertical flipping
  fill_mode: "nearest"      # Filling mode for rotations/shifts
  
  # Advanced augmentations
  color_jitter: true        # Enable color space augmentations
  gaussian_noise: true      # Add random Gaussian noise
  noise_stddev: 0.01        # Standard deviation for noise
  random_erasing: true      # Random occlusions (10% probability)
  erasing_prob: 0.1         # Probability of applying random erasing
  perspective_transform: true # Enable perspective transformations
  perspective_delta: 0.1    # Max distortion for perspective transform
  
  # Batch-level augmentations
  batch_augmentations: true # Enable batch-level augmentations
  mixup: true               # Enable MixUp augmentation
  mixup_alpha: 0.2          # Alpha parameter for MixUp
  cutmix: true              # Enable CutMix augmentation
  cutmix_alpha: 1.0         # Alpha parameter for CutMix
  
  # Validation transforms
  validation_augmentation: true  # Enable validation-time processing
  validation_resize_factor: 1.14 # Resize factor before center crop
```

#### Data Validation

```yaml
data_validation:
  enabled: true
  min_samples_per_class: 5      # Minimum samples per class
  max_class_imbalance_ratio: 10.0 # Max class imbalance ratio
  min_image_dimensions: [32, 32] # Minimum image dimensions
  check_corrupt_images: true     # Verify image integrity
  check_duplicates: false        # Check for duplicate images
  max_workers: 16                # Parallelism for validation
  max_images_to_check: 10000     # Limits validation time
```

#### Model Optimization

##### Quantization

```yaml
quantization:
  enabled: false
  type: "post_training"     # Options: "post_training", "during_training"
  format: "int8"            # Options: "int8", "float16"
  optimize_for_inference: true  # Apply graph optimizations
  measure_performance: true     # Evaluate after quantization
```

##### Pruning

```yaml
pruning:
  enabled: false
  type: "magnitude"         # Options: "magnitude", "structured"
  target_sparsity: 0.5      # Percentage of weights to prune
  during_training: true     # Apply pruning schedule during training
  schedule: "polynomial"    # Options: "constant", "polynomial"
  start_step: 0             # Training step to begin pruning
  end_step: 100             # Training step to reach target sparsity
  frequency: 10             # Apply pruning every N steps
```

### Model-Specific Configuration

DeepCropDX supports model-specific configurations in `models.yaml` that override global settings for particular architectures:

#### Base Model Configuration

```yaml
ResNet50:
  input_shape: [224, 224, 3]
  weights: "imagenet"     # Use ImageNet weights
  include_top: false      # Exclude classification layer
  pooling: avg            # Options: avg, max, None
  dropout_rate: 0.2       # Dropout rate before classification
  attention_type: null    # Options: "se", "cbam", "spatial", null
```

#### Fine-Tuning Configuration

```yaml
fine_tuning:
  enabled: true             # Enable fine-tuning
  freeze_layers: 50         # Number of layers to freeze
  progressive: true         # Gradually unfreeze layers
  finetuning_epochs: 5      # Epochs for fine-tuning
```

#### Preprocessing Configuration

```yaml
preprocessing:
  rescale: 0.00392156862745098  # 1/255 as float
  validation_augmentation: false # Model-specific validation augmentation
```

#### Model-Specific Hyperparameters

```yaml
hyperparameters:
  learning_rate: 0.0005     # Override global learning rate
  batch_size: 32            # Override global batch size
  optimizer: "adam"         # Override global optimizer
  
  # Discriminative learning rates
  discriminative_lr:
    enabled: true           # Enable discriminative learning rates
    base_lr: 0.0003         # Base learning rate (last layer)
    factor: 0.3             # Reduction factor for earlier layers
```

## Performance Considerations

### Quantization Performance

- **INT8 Quantization**:
  - Size reduction: 75% (4x smaller)
  - Speed improvement: 2-3x on compatible hardware
  - Accuracy impact: Typically 0-1% loss

- **Float16 Quantization**:
  - Size reduction: 50% (2x smaller)
  - Speed improvement: 1.5-2x on compatible hardware
  - Accuracy impact: Negligible (often <0.5%)

### Pruning Performance

- **Magnitude Pruning (50% sparsity)**:
  - Size reduction: 50% (2x smaller)
  - Speed improvement: Hardware-dependent (0-30%)
  - Accuracy impact: 0-1% loss

- **Structured Pruning (50% channels)**:
  - Size reduction: 50% (2x smaller)
  - Speed improvement: 1.5-2x on most hardware
  - Accuracy impact: 1-3% loss

### Learning Rate Scheduling Performance

- **Warmup Scheduling**:
  - Training stability improvement: Significant with large batch sizes
  - Convergence speed: Similar or slightly better
  - Final accuracy improvement: 0-1%

- **One-Cycle Policy**:
  - Convergence speed: 20-30% faster
  - Final accuracy improvement: 0.5-2%

### Attention Mechanisms Performance

- **Squeeze-and-Excitation**:
  - Computational overhead: 2-5%
  - Accuracy improvement: 1-3%

- **CBAM**:
  - Computational overhead: 5-10%
  - Accuracy improvement: 2-4%

- **Spatial Attention**:
  - Computational overhead: 3-7%
  - Accuracy improvement: 1-3%

### Hardware Support

Different optimizations work better on different hardware:

- **INT8 Quantization**: Well-supported on most modern hardware (CPUs, GPUs, TPUs, Edge devices)
- **Float16 Quantization**: Best on GPUs with native FP16 support (NVIDIA Volta+ architectures)
- **Pruning**: Benefits dependent on hardware support for sparse operations
- **Attention Mechanisms**: Generally scale well across all hardware types

### Combined Optimization Strategy

For optimal results in resource-constrained environments, a recommended strategy is:

1. **Training Phase**:
   - Use learning rate finder to determine optimal learning rate
   - Apply one-cycle learning rate policy
   - Add appropriate attention mechanism (SE for efficiency, CBAM for accuracy)
   - Apply gradual pruning during training

2. **Deployment Phase**:
   - Apply post-training quantization (INT8 for edge devices, FP16 for server GPUs)
   - Strip pruning masks for deployment
   - Optimize model graph for inference

This combined approach typically yields models that are 4-8x smaller and 2-3x faster with minimal accuracy degradation (<2%).

Always test performance on your target deployment hardware and validate accuracy on your specific dataset to ensure the optimizations provide the desired benefits without unacceptable accuracy degradation.

## Best Practices & Recommendations

1. **Start with `lr_finder: enabled: true`**:  
   For new datasets or model architectures, always enable the learning rate finder to determine a suitable learning rate range before full training.

2. **Use `validation_augmentation: true` for small datasets**:  
   Enabling validation-time augmentation, especially for models like `EfficientNetB0_SE` and `ResNet_CBAM`, can improve validation stability and generalization, particularly when data is limited.

3. **Experiment with `EfficientNetB0_SE`**:  
   This model provides a good balance between accuracy and computational efficiency and is a recommended starting point.

4. **Adjust `image_size` based on model and resource constraints**:  
   For resource-constrained environments or faster prototyping, consider reducing `image_size`. For models like `EfficientNetB1` and above, larger input sizes (`240x240`, `260x260`, etc.) are generally recommended.

5. **Monitor Hardware Usage**:  
   Regularly check hardware utilization (CPU, GPU, Memory) using TensorBoard or system monitoring tools to optimize resource usage and identify bottlenecks.

6. **Architecture-Specific Notes**:
   - **MobileNet Variants**: Use smaller input sizes for mobile deployment with lower dropout rates (`0.1-0.2`)
   - **EfficientNet_SE & ResNet_CBAM**: These incorporate attention mechanisms for improved feature extraction and typically benefit from custom learning rates (`0.0002-0.0005`) for fine-tuning
   - **ResNet_CBAM**: Enable `discriminative_lr` to apply different learning rates to different parts of ResNet architectures, optimizing fine-tuning
