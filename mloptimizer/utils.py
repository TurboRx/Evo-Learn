import time
import logging
from typing import Callable, Any

# Configure the logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def timer(unit: str = "seconds", log_level: int = logging.INFO) -> Callable:
    """
    Advanced decorator to measure the execution time of a function.

    Args:
        unit (str): Unit of time to display ('seconds' or 'milliseconds'). Default is 'seconds'.
        log_level (int): Logging level for the timing information. Default is logging.INFO.

    Returns:
        Callable: A decorator for measuring execution time.
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000 if unit == "milliseconds" else (end_time - start_time)
                unit_label = "ms" if unit == "milliseconds" else "s"

                logger.log(log_level, f"Function '{func.__name__}' executed in {elapsed_time:.2f} {unit_label}")
                logger.log(log_level, f"Arguments: args={args}, kwargs={kwargs}")

                return result
            except Exception as e:
                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000 if unit == "milliseconds" else (end_time - start_time)
                unit_label = "ms" if unit == "milliseconds" else "s"

                logger.error(f"Function '{func.__name__}' failed after {elapsed_time:.2f} {unit_label}")
                logger.exception(e)
                raise
        return wrapper
    return decorator
