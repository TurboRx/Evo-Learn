# mloptimizer/core.py

import pandas as pd
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier
from sklearn.metrics import accuracy_score

def run_automl(data_path, target_column, generations=5, population_size=20, test_size=0.2, random_state=42):
    """
    Runs automated machine learning using TPOT.

    Args:
        data_path (str): Path to the dataset CSV file.
        target_column (str): Name of the target variable column.
        generations (int): Number of generations for TPOT.
        population_size (int): Population size for TPOT.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generation.

    Returns:
        float: Accuracy of the best model on the test set.
    """
    data = pd.read_csv(data_path)
    data = data.dropna()
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    tpot = TPOTClassifier(generations=generations, population_size=population_size, verbosity=2, random_state=random_state)
    tpot.fit(X_train, y_train)

    y_pred = tpot.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy}')

    tpot.export('mloptimizer/models/best_model.py')
    print("The best model has been exported to mloptimizer/models/best_model.py")
    return accuracy
