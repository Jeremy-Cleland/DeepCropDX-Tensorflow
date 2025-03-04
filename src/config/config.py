import os
from pathlib import Path


class ProjectPaths:
    def __init__(self, base_dir=None):
        """Initialize project paths.

        Args:
            base_dir: Base directory of the project. If None, uses the parent directory of this file.
        """
        if base_dir is None:
            # Get the absolute path of the parent directory
            self.base_dir = Path(__file__).parent.parent.parent.absolute()
        else:
            self.base_dir = Path(base_dir).absolute()

        # Source code directories
        self.src_dir = self.base_dir / "src"
        self.config_dir = self.src_dir / "config"
        self.model_configs_dir = self.config_dir / "model_configs"
        self.models_dir = self.src_dir / "models"
        self.preprocessing_dir = self.src_dir / "preprocessing"
        self.evaluation_dir = self.src_dir / "evaluation"
        self.training_dir = self.src_dir / "training"
        self.utils_dir = self.src_dir / "utils"
        self.scripts_dir = self.src_dir / "scripts"

        # Data directories
        self.data_dir = self.base_dir / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"

        # Model output directories
        self.trials_dir = self.base_dir / "trials"

        # Logs directory
        self.logs_dir = self.base_dir / "logs"

        # Ensure critical directories exist
        self._ensure_directories_exist()

    def _ensure_directories_exist(self):
        """Create all necessary directories if they don't exist"""
        directories = [
            self.src_dir,
            self.config_dir,
            self.model_configs_dir,
            self.models_dir,
            self.preprocessing_dir,
            self.evaluation_dir,
            self.training_dir,
            self.utils_dir,
            self.scripts_dir,
            self.data_dir,
            self.raw_data_dir,
            self.processed_data_dir,
            self.trials_dir,
            self.logs_dir,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_model_trial_dir(self, model_name, run_id=None):
        """Get the trial directory for a specific model.

        Args:
            model_name: Name of the model (e.g., "EfficientNetB1")
            run_id: Specific run ID. If None, will use a timestamp.

        Returns:
            Path to the model trial directory
        """
        from datetime import datetime

        model_dir = self.trials_dir / model_name

        if run_id is None:
            # Generate a timestamped run ID
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Find the latest run number
            existing_runs = [
                d for d in model_dir.glob(f"run_{timestamp}_*") if d.is_dir()
            ]
            if existing_runs:
                latest_num = max([int(d.name.split("_")[-1]) for d in existing_runs])
                run_id = f"run_{timestamp}_{(latest_num + 1):03d}"
            else:
                run_id = f"run_{timestamp}_001"

        run_dir = model_dir / run_id

        # Create subdirectories for training and evaluation
        train_dir = run_dir / "training"
        eval_dir = run_dir / "evaluation"
        checkpoints_dir = train_dir / "checkpoints"
        plots_dir = train_dir / "plots"
        tensorboard_dir = train_dir / "tensorboard"

        # Create all directories
        for directory in [
            run_dir,
            train_dir,
            eval_dir,
            checkpoints_dir,
            plots_dir,
            tensorboard_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        return run_dir

    def get_config_path(self):
        """Get the path to the main configuration file"""
        return self.config_dir / "config.yaml"

    def get_model_config_path(self, model_name=None):
        """Get the path to the model configuration file.

        Args:
            model_name: Name of the model. If None, returns the models.yaml path.

        Returns:
            Path to the model configuration file
        """
        if model_name is None:
            return self.model_configs_dir / "models.yaml"

        # Try model-specific file first
        model_file = f"{model_name.lower().split('_')[0]}.yaml"
        specific_path = self.model_configs_dir / model_file

        if specific_path.exists():
            return specific_path

        # Fall back to models.yaml
        return self.model_configs_dir / "models.yaml"


# Create a singleton instance
project_paths = ProjectPaths()


def get_paths():
    """Get the project paths singleton instance"""
    return project_paths
