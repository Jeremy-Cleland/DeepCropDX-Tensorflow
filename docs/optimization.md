# Model Optimization

This document provides details on the various model optimization techniques available in this project, including:

- Quantization
- Pruning
- Learning Rate Scheduling with Warmup
- Other optimization techniques

## Model Quantization

Model quantization reduces model size and improves inference speed by representing weights with lower precision.

### Configuration

To enable quantization, add the following to your config:

```yaml
training:
  quantization:
    enabled: true
    # Quantization type: "post_training" or "during_training"
    type: "post_training"
    # Quantization format: "int8", "float16", etc.
    format: "int8"
    # Whether to optimize for inference
    optimize_for_inference: true
    # Whether to measure performance after quantization
    measure_performance: true
```

### Supported Quantization Types

1. **Post-Training Quantization**:
   - Applied after model training
   - Minimal accuracy impact
   - Supports int8, float16 formats

2. **During-Training Quantization**:
   - Trains model with simulated quantization effects
   - Better accuracy for heavily quantized models
   - Requires longer training time

## Model Pruning

Pruning removes redundant weights to create sparse models that are smaller and potentially faster.

### Configuration

```yaml
training:
  pruning:
    enabled: true
    # Pruning type: "magnitude", "structured", etc.
    type: "magnitude"
    # Target sparsity (percentage of weights to prune)
    target_sparsity: 0.5
    # Whether to perform pruning during training
    during_training: true
    # Pruning schedule: "constant" or "polynomial"
    schedule: "polynomial"
    # Start step for pruning
    start_step: 0
    # End step for pruning
    end_step: 100
    # Pruning frequency (every N steps)
    frequency: 10
```

### Pruning Types

1. **Magnitude Pruning**:
   - Removes weights based on their absolute magnitude
   - Most common pruning technique

2. **Structured Pruning**:
   - Prunes entire structures (channels, filters)
   - More hardware-friendly but may impact accuracy more

## Learning Rate Schedulers

Learning rate scheduling can significantly improve model convergence and final accuracy.

### Warmup Scheduling

Warmup gradually increases the learning rate from a small value to the base learning rate over a set number of epochs, helping with training stability.

```yaml
training:
  lr_schedule:
    enabled: true
    # Type can be: warmup_cosine, warmup_exponential, warmup_step, one_cycle
    type: "warmup_cosine"
    # Number of warmup epochs
    warmup_epochs: 5
    # Minimum learning rate
    min_lr: 1.0e-6
```

### One-Cycle Policy

The one-cycle policy rapidly increases the learning rate and then gradually decreases it, often leading to faster convergence.

```yaml
training:
  lr_schedule:
    enabled: true
    type: "one_cycle"
    # Maximum learning rate
    max_lr: 0.01
    # Factor to determine initial learning rate (max_lr / div_factor)
    div_factor: 25.0
    # Percentage of cycle spent increasing learning rate
    pct_start: 0.3
    # Minimum learning rate
    min_lr: 1.0e-6
```

## Usage in Code

To apply these optimizations in your code:

```python
from src.models.model_optimizer import ModelOptimizer
from src.training.learning_rate_scheduler import get_warmup_scheduler

# Create and apply optimizer
optimizer = ModelOptimizer(config)
quantized_model = optimizer.apply_quantization(model, representative_dataset)

# Get learning rate scheduler
lr_scheduler = get_warmup_scheduler(config)
callbacks.append(lr_scheduler)
```

## Performance Considerations

- **Quantization** typically offers 2-4x size reduction with minimal accuracy loss
- **Pruning** can reduce model size by 3-10x but may require retraining
- **Learning rate scheduling** typically improves accuracy by 1-3% and can reduce training time
- Combined optimizations provide cumulative benefits

## Hardware Support

Different optimizations work better on different hardware:

- **INT8 Quantization**: Well-supported on most modern hardware (CPUs, GPUs, TPUs)
- **Float16 Quantization**: Best on GPUs with FP16 support
- **Pruning**: Benefits dependent on hardware support for sparse operations

Always test performance on your target deployment hardware.