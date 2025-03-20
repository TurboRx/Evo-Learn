# mloptimizer/utils.py

import time

def timer(func):
    """
    Decorator to measure the execution time of a function.

    Args:
        func (callable): Function to be timed.

    Returns:
        callable: Wrapped function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper
