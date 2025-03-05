"""
Memory management utilities to prevent memory leaks during training.
"""

import gc
import os
import sys
import psutil
import logging
from typing import Dict, Any, Optional, Callable

import tensorflow as tf
import numpy as np

logger = logging.getLogger("plant_disease_detection")


def get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dictionary with memory usage statistics
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    # Get system memory info
    system_memory = psutil.virtual_memory()
    
    memory_stats = {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        "percent_used": process.memory_percent(),
        "system_total_gb": system_memory.total / (1024 * 1024 * 1024),
        "system_available_gb": system_memory.available / (1024 * 1024 * 1024),
        "system_percent": system_memory.percent
    }
    
    return memory_stats


def log_memory_usage(step: int = 0, prefix: str = "") -> Dict[str, Any]:
    """
    Log current memory usage.
    
    Args:
        step: Current step (for logging)
        prefix: Prefix for log message
        
    Returns:
        Memory usage statistics
    """
    memory_stats = get_memory_usage()
    
    log_message = f"{prefix}Memory usage: "
    log_message += f"RSS: {memory_stats['rss_mb']:.1f}MB, "
    log_message += f"Process: {memory_stats['percent_used']:.1f}%, "
    log_message += f"System: {memory_stats['system_percent']:.1f}%"
    
    logger.info(log_message)
    return memory_stats


def clean_memory(clean_gpu: bool = True) -> None:
    """
    Clean up memory resources and force garbage collection.
    
    Args:
        clean_gpu: Whether to clear GPU memory as well
    """
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Additional TensorFlow cleanup
    if hasattr(tf.keras.backend, 'set_session'):
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session())
    
    # Clean GPU memory if requested and available
    if clean_gpu and tf.config.list_physical_devices('GPU'):
        try:
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.reset_memory_stats(gpu)
        except Exception as e:
            logger.warning(f"Failed to reset GPU memory stats: {e}")


def memory_monitoring_decorator(
    func: Callable,
    log_prefix: str = "",
    log_interval: int = 1
) -> Callable:
    """
    Decorator to monitor memory usage during a function execution.
    
    Args:
        func: Function to decorate
        log_prefix: Prefix for log messages
        log_interval: Interval for memory logging (in seconds)
        
    Returns:
        Decorated function
    """
    from functools import wraps
    import time
    import threading
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        stop_monitor = threading.Event()
        memory_stats = []
        
        def monitor_memory():
            start_time = time.time()
            while not stop_monitor.is_set():
                try:
                    stats = get_memory_usage()
                    stats["time"] = time.time() - start_time
                    memory_stats.append(stats)
                    
                    # Log current memory usage
                    log_memory_usage(step=len(memory_stats), prefix=log_prefix)
                    
                    # Wait for next interval
                    time.sleep(log_interval)
                except Exception as e:
                    logger.error(f"Error in memory monitoring: {e}")
                    break
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        try:
            # Call the original function
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop monitoring
            stop_monitor.set()
            monitor_thread.join(timeout=1.0)
            
            # Log final memory usage
            final_stats = log_memory_usage(prefix=f"{log_prefix}Final ")
            
            # Clean up resources
            clean_memory()
    
    return wrapper


def limit_gpu_memory_growth() -> None:
    """
    Configure TensorFlow to limit GPU memory growth to prevent OOM errors.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                
            logger.info(f"GPU memory growth enabled for {len(gpus)} GPUs")
        except RuntimeError as e:
            logger.error(f"Error setting GPU memory growth: {e}")