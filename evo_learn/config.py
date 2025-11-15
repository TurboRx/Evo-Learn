"""Configuration management with validation."""

from typing import Dict, Any, Optional
from pathlib import Path
import yaml
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvoLearnConfig:
    """Configuration dataclass for Evo-Learn."""
    
    # Task settings
    default_task: str = "classification"
    random_state: int = 42
    test_size: float = 0.2
    
    # TPOT parameters
    generations: int = 5
    population_size: int = 20
    max_time_mins: Optional[int] = None
    max_eval_time_mins: Optional[int] = 5
    
    # Preprocessing
    handle_categoricals: bool = True
    impute_strategy: str = "median"
    scale_numeric: bool = True
    
    # Model selection
    baseline: bool = False
    
    # Output
    output_dir: str = "mloptimizer/models"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    verbose: bool = False
    
    # Cross-validation
    cv_folds: int = 5
    use_cv: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """Validate configuration values.
        
        Raises:
            ValueError: If any configuration value is invalid
        """
        # Validate task
        if self.default_task not in ["classification", "regression"]:
            raise ValueError(
                f"default_task must be 'classification' or 'regression', "
                f"got '{self.default_task}'"
            )
        
        # Validate test_size
        if not 0.0 < self.test_size < 1.0:
            raise ValueError(
                f"test_size must be between 0 and 1, got {self.test_size}"
            )
        
        # Validate generations and population_size
        if self.generations < 1:
            raise ValueError(f"generations must be >= 1, got {self.generations}")
        
        if self.population_size < 1:
            raise ValueError(
                f"population_size must be >= 1, got {self.population_size}"
            )
        
        # Validate impute_strategy
        valid_strategies = ["mean", "median", "most_frequent", "constant"]
        if self.impute_strategy not in valid_strategies:
            raise ValueError(
                f"impute_strategy must be one of {valid_strategies}, "
                f"got '{self.impute_strategy}'"
            )
        
        # Validate cv_folds
        if self.cv_folds < 2:
            raise ValueError(f"cv_folds must be >= 2, got {self.cv_folds}")
        
        # Validate log_level
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, "
                f"got '{self.log_level}'"
            )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "EvoLearnConfig":
        """Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            EvoLearnConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
        
        logger.info(f"Loaded configuration from: {config_path}")
        return cls(**config_dict)
    
    def to_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            output_path: Path to save YAML file
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration saved to: {output_path}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of config
        """
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def update(self, **kwargs) -> None:
        """Update configuration with new values.
        
        Args:
            **kwargs: Key-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")
        
        # Re-validate after update
        self.validate()


def load_config(config_path: Optional[str] = None) -> EvoLearnConfig:
    """Load configuration from file or use defaults.
    
    Args:
        config_path: Optional path to YAML config file
        
    Returns:
        EvoLearnConfig instance
    """
    if config_path:
        try:
            return EvoLearnConfig.from_yaml(config_path)
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            logger.warning("Using default configuration")
    
    return EvoLearnConfig()
