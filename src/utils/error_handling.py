"""
Error handling utilities for graceful error handling throughout the codebase.
"""

import sys
import traceback
import logging
from typing import Optional, Callable, Any, Dict, Type, Union
from functools import wraps

# Configure logger
logger = logging.getLogger("plant_disease_detection")


class DataError(Exception):
    """Exception raised for errors in data loading and processing."""
    pass


class ModelError(Exception):
    """Exception raised for errors in model creation and training."""
    pass


class ConfigError(Exception):
    """Exception raised for errors in configuration."""
    pass


def handle_exception(
    exc: Exception, 
    error_msg: str, 
    log_traceback: bool = True
) -> None:
    """
    Handle exceptions with consistent logging and user feedback.
    
    Args:
        exc: The exception that was raised
        error_msg: A human-readable error message
        log_traceback: Whether to log the full traceback
    """
    # Log the error with appropriate level
    if isinstance(exc, (ValueError, KeyError, TypeError)):
        logger.error(f"{error_msg}: {str(exc)}")
    else:
        logger.critical(f"{error_msg}: {str(exc)}")
    
    # Log traceback for unexpected errors
    if log_traceback:
        logger.debug(traceback.format_exc())
    
    # Print user-friendly message to console
    print(f"Error: {error_msg}")
    print(f"Details: {str(exc)}")


def try_except_decorator(
    error_msg: str,
    exception_types: Union[Type[Exception], tuple] = Exception,
    cleanup_func: Optional[Callable] = None
) -> Callable:
    """
    Decorator for try-except error handling pattern.
    
    Args:
        error_msg: Message to show when an exception occurs
        exception_types: Type(s) of exceptions to catch
        cleanup_func: Optional function to call in finally block
        
    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except exception_types as e:
                handle_exception(e, error_msg)
                raise
            finally:
                if cleanup_func:
                    cleanup_func()
        return wrapper
    return decorator


def retry_on_error(
    max_retries: int = 3,
    retry_exceptions: Union[Type[Exception], tuple] = Exception,
    backoff_factor: float = 2.0
) -> Callable:
    """
    Decorator that retries a function on specific exceptions.
    
    Args:
        max_retries: Maximum number of retry attempts
        retry_exceptions: Exception types to retry on
        backoff_factor: Factor to multiply delay between retries
        
    Returns:
        Decorated function with retry capabilities
    """
    import time
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            delay = 1.0
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Maximum retries ({max_retries}) reached for {func.__name__}")
                        raise
                    
                    wait_time = delay * (backoff_factor ** (retries - 1))
                    logger.warning(
                        f"Retry {retries}/{max_retries} for {func.__name__} "
                        f"after {wait_time:.1f}s due to: {str(e)}"
                    )
                    time.sleep(wait_time)
            
            # This should never be reached
            raise RuntimeError("Unexpected end of retry loop")
            
        return wrapper
    return decorator