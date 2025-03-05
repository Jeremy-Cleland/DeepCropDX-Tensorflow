"""
CLI utilities for handling command-line arguments and configuration loading.
"""

from typing import Tuple, Dict, Any, List, Optional

from src.config.config_manager import ConfigManager


def handle_cli_args() -> Tuple[ConfigManager, Dict[str, Any], bool]:
    """
    Parse command-line arguments and load configuration.
    
    Returns:
        Tuple containing:
            - ConfigManager: Initialized configuration manager
            - Dict: Loaded configuration
            - bool: Whether to print hardware summary and exit
    """
    # Set up configuration manager
    config_manager = ConfigManager()
    args = config_manager.parse_args()
    
    # Check if we should just print hardware summary
    should_print_hardware = config_manager.should_print_hardware_summary()
    
    # Load configuration with command-line overrides
    config = config_manager.load_config()
    
    return config_manager, config, should_print_hardware


def get_project_info(config: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract project name and version from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple containing project name and version
    """
    project_info = config.get("project", {})
    project_name = project_info.get("name", "Plant Disease Detection")
    project_version = project_info.get("version", "1.0.0")
    
    return project_name, project_version