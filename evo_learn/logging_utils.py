"""Enhanced logging utilities with progress tracking."""

import logging
import sys
from typing import Optional
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import colorama
from colorama import Fore, Style

colorama.init(autoreset=True)


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with color."""
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    verbose: bool = False,
) -> logging.Logger:
    """Set up logging with optional file output and colored console.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        verbose: If True, use DEBUG level and detailed format
        
    Returns:
        Configured logger instance
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    if verbose:
        numeric_level = logging.DEBUG
    
    # Create logger
    logger = logging.getLogger("evo_learn")
    logger.setLevel(numeric_level)
    logger.handlers.clear()  # Remove existing handlers
    
    # Console handler with color
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if verbose:
        console_format = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
    else:
        console_format = "%(asctime)s | %(levelname)s | %(message)s"
    
    console_formatter = ColoredFormatter(
        console_format,
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (no color)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        
        file_format = "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
        file_formatter = logging.Formatter(
            file_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")
    
    return logger


class ProgressTracker:
    """Track and display progress for long-running operations."""
    
    def __init__(self, total: int, desc: str = "Processing", disable: bool = False):
        """Initialize progress tracker.
        
        Args:
            total: Total number of steps
            desc: Description to display
            disable: If True, disable progress bar
        """
        self.pbar = tqdm(
            total=total,
            desc=desc,
            disable=disable,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        )
        self.start_time = datetime.now()
    
    def update(self, n: int = 1, status: Optional[str] = None) -> None:
        """Update progress.
        
        Args:
            n: Number of steps to increment
            status: Optional status message to display
        """
        self.pbar.update(n)
        if status:
            self.pbar.set_postfix_str(status)
    
    def close(self) -> None:
        """Close progress bar."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.pbar.close()
        return elapsed
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def log_experiment_start(
    logger: logging.Logger,
    task: str,
    data_shape: tuple,
    config: dict
) -> None:
    """Log experiment start with configuration details.
    
    Args:
        logger: Logger instance
        task: Task type (classification/regression)
        data_shape: Shape of input data (rows, cols)
        config: Configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("Starting Evo-Learn AutoML Experiment")
    logger.info("=" * 60)
    logger.info(f"Task: {task}")
    logger.info(f"Data shape: {data_shape[0]} samples, {data_shape[1]} features")
    logger.info(f"Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)


def log_experiment_end(
    logger: logging.Logger,
    duration: float,
    metrics: dict,
    model_path: str
) -> None:
    """Log experiment completion with results.
    
    Args:
        logger: Logger instance
        duration: Total duration in seconds
        metrics: Dictionary of evaluation metrics
        model_path: Path to saved model
    """
    logger.info("=" * 60)
    logger.info("Experiment Completed Successfully")
    logger.info("=" * 60)
    logger.info(f"Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    logger.info(f"Metrics:")
    for metric_name, metric_value in metrics.items():
        logger.info(f"  {metric_name}: {metric_value:.4f}")
    logger.info(f"Model saved: {model_path}")
    logger.info("=" * 60)
