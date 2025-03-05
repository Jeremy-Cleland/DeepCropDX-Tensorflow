# DeepCropDX: Advanced Plant Disease Detection System

![Plant Disease Detection](https://img.shields.io/badge/DeepCropDX-Plant%20Disease%20Detection-brightgreen)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16+-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

DeepCropDX is a state-of-the-art deep learning system for accurate detection and classification of plant diseases from images. Leveraging cutting-edge computer vision techniques and neural network architectures, this platform provides agricultural researchers, farmers, and technologists with a powerful tool to identify plant diseases early and accurately, potentially saving crops and increasing yields.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Advanced Features](#advanced-features)
- [Project Structure](#project-structure)
- [Model Architectures](#model-architectures)
- [Data Processing](#data-processing)
- [Training Pipeline](#training-pipeline)
- [Evaluation & Metrics](#evaluation--metrics)
- [Experiment Tracking](#experiment-tracking)
- [Optimization Techniques](#optimization-techniques)
- [Contributing](#contributing)
- [License](#license)

## Overview

DeepCropDX combines multiple state-of-the-art neural network architectures with advanced data augmentation techniques to create a comprehensive plant disease detection system. The platform supports various model architectures (ResNet, DenseNet, EfficientNet, MobileNet, Vision Transformers, etc.) and provides extensive tools for training, evaluation, and deployment.

### Key Benefits

- **High Accuracy**: Achieves state-of-the-art performance across multiple plant disease datasets
- **Flexibility**: Supports numerous model architectures and customization options
- **Efficiency**: Optimized data pipeline and training process
- **Reproducibility**: Comprehensive experiment tracking and model registry
- **Scalability**: Designed to handle large datasets and complex models

## Features

### Core Features

- **Comprehensive Model Support**: Wide range of neural network architectures
- **Advanced Data Augmentation**: State-of-the-art techniques for improved generalization
- **Flexible Configuration System**: YAML-based configuration with command-line overrides
- **Robust Evaluation Metrics**: Detailed performance analysis and visualization
- **Model Registry**: Systematic tracking of all training runs and model performance
- **Hardware Optimization**: Efficient use of GPU/CPU resources with mixed precision training

### Advanced Capabilities

- **Attention Mechanisms**: Squeeze-and-Excitation (SE), CBAM, and Spatial Attention
- **Advanced Learning Rate Scheduling**: Warmup, cosine decay, one-cycle policy
- **Model Optimization**: Quantization and pruning for efficient deployment
- **Batch-level Augmentations**: MixUp and CutMix for enhanced training
- **Memory Management**: Prevents memory leaks during extended training sessions
- **Detailed Reporting**: Generates comprehensive HTML reports with visualizations

## System Architecture

DeepCropDX is built with a modular architecture that separates concerns and promotes maintainability:

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│ Configuration   │────▶│ Data Pipeline    │────▶│ Model Factory  │
│ Management      │     │                  │     │                │
└─────────────────┘     └──────────────────┘     └────────────────┘
                                                         │
┌─────────────────┐     ┌──────────────────┐            ▼
│ Model Registry  │◀────│ Training         │◀───────────┘
│ & Reporting     │     │ Pipeline         │
└─────────────────┘     └──────────────────┘
        ▲                        │
        │                        ▼
        │               ┌──────────────────┐
        └───────────────│ Evaluation       │
                        │ System           │
                        └──────────────────┘
```

Each component is designed to be independently testable, maintainable, and extensible.

## Installation

### Prerequisites

- Python 3.10+
- TensorFlow 2.16+
- CUDA-compatible GPU (recommended)

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/deepcropdx.git
   cd deepcropdx
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Configure GPU settings in `src/config/config.yaml`

## Usage

### Basic Training

```bash
# Train a single model
python -m src.main --model ResNet50 --data_dir data/processed

# Train multiple specific models
python -m src.main --models EfficientNetB0 MobileNetV2 --data_dir data/processed

# Train all models defined in the configuration
python -m src.main --all_models --data_dir data/processed
```

### Advanced Training Options

```bash
# Use a custom configuration file
python -m src.main --model ResNet50 --config custom_config.yaml

# Enable learning rate warmup
python -m src.main --model ResNet50 --warmup_epochs 5

# Resume training from checkpoints
python -m src.main --model ResNet50 --resume

# Find optimal learning rate before training
python -m src.main --model ResNet50 --find_lr
```

### Evaluation

```bash
# Evaluate a trained model
python -m src.scripts.evaluate --model ResNet50 --run_id latest

# Compare multiple models
python -m src.scripts.compare_models --models ResNet50 EfficientNetB0 MobileNetV2
```

### Model Registry Interface

```bash
# List all trained models
python -m src.scripts.registry_cli list

# View details for a specific model
python -m src.scripts.registry_cli details --model ResNet50

# Generate HTML report of all models
python -m src.scripts.registry_cli report
```

## Advanced Features

### Learning Rate Scheduling

DeepCropDX implements sophisticated learning rate scheduling techniques:

```yaml
# Example learning rate schedule configuration
training:
  lr_schedule:
    enabled: true
    type: "warmup_cosine"
    warmup_epochs: 5
    min_lr: 1.0e-6
```

Available scheduling types:

- `warmup_cosine`: Linear warmup followed by cosine decay
- `warmup_exponential`: Linear warmup followed by exponential decay
- `warmup_step`: Linear warmup followed by step decay
- `one_cycle`: One-cycle learning rate policy (fast increase, then decrease)

### Model Optimization

#### Quantization

```bash
# Enable post-training quantization
python -m src.main --model MobileNetV2 --quantize
```

Quantization reduces model size and improves inference speed by representing weights with lower precision.

#### Pruning

```bash
# Enable model pruning
python -m src.main --model ResNet50 --pruning
```

Pruning removes redundant connections in the network, creating sparse models that are smaller and potentially faster.

### Attention Mechanisms

```bash
# Add Squeeze-and-Excitation attention to a model
python -m src.main --model ResNet50 --attention se

# Add CBAM attention to a model
python -m src.main --model ResNet50 --attention cbam

# Add Spatial attention to a model
python -m src.main --model ResNet50 --attention spatial
```

Attention mechanisms help models focus on the most relevant parts of the image, improving accuracy for complex patterns.

## Project Structure

```
root/
├── src/                      # Source code
│   ├── config/               # Configuration management
│   ├── evaluation/           # Evaluation utilities
│   ├── models/               # Model definitions
│   ├── preprocessing/        # Data preprocessing
│   ├── scripts/              # Executable scripts
│   ├── training/             # Training utilities
│   ├── utils/                # Utility functions
│   └── main.py               # Main entry point
├── data/                     # Data storage
│   ├── raw/                  # Raw data
│   └── processed/            # Processed data
├── trials/                   # Model training results
│   ├── registry.json         # Model registry database
│   └── [ModelName]/          # Results for specific models
├── docs/                     # Documentation
└── config/                   # Configuration examples
    └── examples/
```

## Model Architectures

DeepCropDX supports a wide range of model architectures:

### Standard CNN Architectures

- ResNet family (50, 101, 152, v2 variants)
- DenseNet family (121, 169, 201)
- MobileNet family (v1, v2, v3)
- EfficientNet family (B0-B7)
- Xception
- InceptionV3/InceptionResNetV2

### Advanced Architectures

- EfficientNetV2 family
- ConvNeXt (tiny, small, base, large)
- Vision Transformers (ViT)

### Custom Attention Mechanisms

- SE (Squeeze and Excitation)
- CBAM (Convolutional Block Attention Module)
- Spatial Attention

Models can be customized with different input sizes, dropout rates, regularization parameters, and pre-trained weights.

## Data Processing

### Data Loading

The system supports various dataset formats and automatically handles splitting into training, validation, and test sets:

```python
# Using the data loader
from src.preprocessing.data_loader import DataLoader

loader = DataLoader(
    data_dir="path/to/dataset",
    validation_split=0.2,
    test_split=0.1,
    seed=42
)

train_data, val_data, test_data = loader.load_dataset()
```

### Data Augmentation Pipeline

DeepCropDX implements a sophisticated data augmentation pipeline:

#### Basic Augmentations

- Random rotations (±20°)
- Random translations (±20%)
- Shear and zoom transformations
- Horizontal and vertical flips

#### Advanced Augmentations

- Color jitter
- Random erasing (occlusion simulation)
- Gaussian noise
- Perspective transformations

#### Batch-level Augmentations

- MixUp: Combines pairs of images and labels
- CutMix: Cuts and pastes patches between images

These augmentations can be configured in the YAML configuration:

```yaml
data_augmentation:
  enabled: true
  rotation_range: 20
  width_shift_range: 0.2
  height_shift_range: 0.2
  zoom_range: 0.2
  horizontal_flip: true
  color_jitter: true
  random_erasing: true
  perspective_transform: true
  batch_augmentations: true
  mixup: true
  cutmix: true
```

## Training Pipeline

The training pipeline is managed by the `Trainer` class, which handles:

- Model compilation and configuration
- Learning rate scheduling
- Callbacks setup (checkpoints, early stopping, TensorBoard)
- Memory management
- Progress tracking
- Model evaluation

### Key Features

- **Flexible Callbacks**: Custom progress bar, TensorBoard integration, etc.
- **Automatic Class Weighting**: Handles class imbalance in datasets
- **Hardware Monitoring**: Tracks GPU usage during training
- **Checkpoint Management**: Saves and loads model checkpoints
- **Gradient Clipping**: Prevents gradient explosion during training

### Batch Training

For training multiple models in sequence:

```python
from src.training.batch_trainer import BatchTrainer

batch_trainer = BatchTrainer(config)
results = batch_trainer.train_models(['ResNet50', 'EfficientNetB0', 'MobileNetV2'])
```

## Evaluation & Metrics

The evaluation system provides comprehensive metrics and visualizations:

### Classification Metrics

- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1-score (macro and weighted)
- ROC AUC (macro and weighted)

### Per-class Metrics

- Class-specific precision/recall/F1
- Support counts

### Visualizations

- Confusion matrices
- ROC curves
- Precision-recall curves
- Loss and accuracy curves

Example visualization code:

```python
from src.evaluation.visualization import plot_confusion_matrix, plot_roc_curve

# Plot confusion matrix
plot_confusion_matrix(y_true, y_pred, class_names, save_path="confusion_matrix.png")

# Plot ROC curves
plot_roc_curve(y_true, y_pred_proba, class_names, save_path="roc_curves.png")
```

## Experiment Tracking

DeepCropDX includes a comprehensive Model Registry for tracking experiments:

### Registry Features

- Tracks all training runs and their performance
- Automatically saves model artifacts and metadata
- Provides tools to compare different models
- Generates reports on model performance

### Experiment Management

```python
from src.model_registry.registry_manager import ModelRegistryManager

# Initialize registry
registry = ModelRegistryManager()

# Get best performing model
best_model = registry.get_model("ResNet50", best=True)

# Compare models
comparison = registry.compare_models(
    model_names=["ResNet50", "EfficientNetB0", "MobileNetV2"],
    metrics=["test_accuracy", "training_time"]
)

# Generate HTML report
registry.generate_report()
```

## Optimization Techniques

### Learning Rate Optimization

- **Warmup Scheduling**: Gradually increases learning rate to prevent early instability
- **Cosine Decay**: Smooth learning rate reduction following cosine curve
- **One-Cycle Policy**: Fast increase then gradual decrease for super-convergence
- **Learning Rate Finder**: Automatically finds optimal learning rates

### Model Optimization

- **Quantization**: Reduces model size and improves inference speed
- **Pruning**: Removes redundant connections for smaller models
- **Mixed Precision Training**: Uses FP16 for faster training
- **Gradient Clipping**: Prevents exploding gradients

### Performance Optimization

- **Memory Management**: Prevents memory leaks during training
- **Efficient Data Pipeline**: Optimized for throughput with prefetching
- **Hardware Utilization**: Makes efficient use of available hardware

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
