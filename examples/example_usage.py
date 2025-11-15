import argparse
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import run_automl
from preprocessing import build_preprocessor
import pandas as pd

def main(data_path, target_column, task, generations, population_size):
    """
    Main function to run AutoML with preprocessing.

    Args:
        data_path (str): Path to the dataset CSV.
        target_column (str): The target column name.
        task (str): 'classification' or 'regression'
        generations (int): Number of generations for AutoML.
        population_size (int): Population size for AutoML.
    """
    try:
        # Run AutoML with automatic preprocessing
        logging.info('Running AutoML with automatic preprocessing...')
        result = run_automl(
            data_path, 
            target_column, 
            task=task,
            generations=generations, 
            population_size=population_size
        )
        
        logging.info(f"AutoML completed successfully!")
        logging.info(f"Metrics: {result['metrics']}")
        logging.info(f"Model saved to: {result['model_path']}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoML on a dataset")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset CSV")
    parser.add_argument('--target_column', type=str, required=True, help="The target column name")
    parser.add_argument('--task', type=str, default='classification', 
                       choices=['classification', 'regression'],
                       help="Machine learning task type")
    parser.add_argument('--generations', type=int, default=5, help="Number of generations for AutoML")
    parser.add_argument('--population_size', type=int, default=20, help="Population size for AutoML")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.data_path, args.target_column, args.task, args.generations, args.population_size)
