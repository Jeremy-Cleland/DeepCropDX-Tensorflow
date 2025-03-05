#!/usr/bin/env python3
"""
Main module for the plant disease detection model training system.
This module provides a command-line interface for training plant disease detection models.
"""

import tensorflow as tf
from typing import Dict, List, Any, Optional

from src.utils.cli_utils import handle_cli_args, get_project_info
from src.utils.hardware_utils import configure_hardware, print_hardware_summary
from src.training.training_pipeline import (
    execute_training_pipeline,
    generate_training_reports,
    clean_up_resources,
)
from src.utils.error_handling import handle_exception
from src.utils.memory_utils import optimize_memory_use


def main() -> int:
    """
    Main entry point for the plant disease detection training system.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    try:
        # Optimize memory at application startup
        optimize_memory_use()
        print("Memory optimized at application startup")

        # Handle command line arguments and load configuration
        config_manager, config, should_print_hardware = handle_cli_args()

        # Print hardware summary and exit if requested
        if should_print_hardware:
            print_hardware_summary()
            return 0

        # Configure hardware
        hardware_info = configure_hardware(config)

        # Get project info and print startup message
        project_name, project_version = get_project_info(config)
        print(f"Starting {project_name} v{project_version} Batch Training")

        # Execute the training pipeline
        batch_trainer, total_time, exit_code = execute_training_pipeline(
            config, config_manager, hardware_info
        )

        # Generate reports if training was successful
        if batch_trainer and exit_code == 0:
            generate_training_reports(batch_trainer, total_time)

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        return 1
    except Exception as e:
        handle_exception(e, "Error in main process")
        return 1
    finally:
        # Always clean up resources at the end
        clean_up_resources()

    return exit_code


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
