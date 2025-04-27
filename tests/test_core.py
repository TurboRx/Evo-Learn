# tests/test_core.py

import unittest
import pandas as pd
from io import StringIO
from mloptimizer.core import run_automl

class TestCore(unittest.TestCase):
    """Unit tests for core functionalities of ML-Optimizer."""

    def setUp(self):
        """Set up a mock dataset in memory."""
        self.mock_data = StringIO("""
        feature1,feature2,target
        1,2,0
        2,3,1
        3,4,0
        4,5,1
        5,6,0
        """)

    def test_run_automl(self):
        """Test the run_automl function for expected output type (accuracy as float)."""
        # Load the mock data into a DataFrame
        data = pd.read_csv(self.mock_data)
        
        # Call run_automl and check return type
        accuracy = run_automl(self.mock_data, 'target', generations=1, population_size=2)
        self.assertIsInstance(accuracy, float)

    def tearDown(self):
        """Clean up resources."""
        self.mock_data.close()

if __name__ == '__main__':
    unittest.main()
    
