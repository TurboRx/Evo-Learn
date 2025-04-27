# tests/test_core.py

import unittest
import pandas as pd
from mloptimizer.core import run_automl

class TestCore(unittest.TestCase):

    def test_run_automl(self):
        # Create a simple test dataset
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0]
        })
        data.to_csv('test_data.csv', index=False)

        # Run AutoML and check if it returns a float (accuracy)
        accuracy = run_automl('test_data.csv', 'target', generations=1, population_size=2) #reduced to make tests faster.
        self.assertIsInstance(accuracy, float)

if __name__ == '__main__':
    unittest.main()
    
