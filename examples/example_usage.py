import argparse
import logging
from mloptimizer.core import run_automl
from mloptimizer.preprocessing import preprocess_data

def main(data_path, target_column, categorical_features, generations, population_size):
    """
    Main function to preprocess data and run AutoML.

    Args:
        data_path (str): Path to the dataset.
        target_column (str): The target column name.
        categorical_features (list): List of categorical features.
        generations (int): Number of generations for AutoML.
        population_size (int): Population size for AutoML.
    """
    try:
        # Preprocess the data
        logging.info('Preprocessing data...')
        preprocessed_data = preprocess_data(data_path, categorical_features)
        preprocessed_data.to_csv('examples/data/preprocessed_data.csv', index=False)
        logging.info('Data preprocessed and saved to examples/data/preprocessed_data.csv')

        # Run AutoML
        logging.info('Running AutoML...')
        accuracy = run_automl('examples/data/preprocessed_data.csv', target_column, generations=generations, population_size=population_size)
        logging.info(f"AutoML Accuracy: {accuracy}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AutoML on a dataset")
    parser.add_argument('--data_path', type=str, required=True, help="Path to the dataset")
    parser.add_argument('--target_column', type=str, required=True, help="The target column name")
    parser.add_argument('--categorical_features', nargs='*', default=[], help="List of categorical features")
    parser.add_argument('--generations', type=int, default=5, help="Number of generations for AutoML")
    parser.add_argument('--population_size', type=int, default=20, help="Population size for AutoML")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    main(args.data_path, args.target_column, args.categorical_features, args.generations, args.population_size)
