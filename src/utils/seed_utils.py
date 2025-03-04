# src/utils/seed_utils.py
import os
import random
import numpy as np
import tensorflow as tf


def set_global_seeds(seed=42):
    """Set all seeds for reproducibility

    Args:
        seed: Integer seed value to use

    Returns:
        The seed value used
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # Try to make TensorFlow deterministic (may not work on all hardware)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

    # Set session config for older TF versions if needed
    try:
        from tensorflow.keras import backend as K

        if hasattr(tf, "ConfigProto"):
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            K.set_session(tf.Session(config=config))
    except:
        pass

    print(f"Global random seeds set to {seed}")
    return seed
