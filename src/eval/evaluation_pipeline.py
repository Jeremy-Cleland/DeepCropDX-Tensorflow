"""
Evaluation pipeline for trained models.
"""

import os
import logging
import tensorflow as tf
from ..utils.memory_utils import optimize_memory_use

logger = logging.getLogger("plant_disease_detection")


def run_evaluation(model_path, data_loader, config):
    """Run evaluation on a trained model"""
    # Optimize memory before evaluation
    optimize_memory_use()
    logger.info(f"Memory optimized before evaluating model: {model_path}")

    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model
    results = model.evaluate(data_loader, verbose=1)

    # Log the results
    logger.info(f"Evaluation results: {results}")

    return results
