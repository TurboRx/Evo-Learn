# tests/test_core.py

import unittest
import pandas as pd
import tempfile
import os
from core import run_automl

class TestCore(unittest.TestCase):
    """Unit tests for core functionalities of Evo-Learn."""

    def setUp(self):
        """Set up a mock dataset in a temporary file."""
        # Create a larger dataset to avoid single-class issues
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            'feature2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
            'target': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def test_run_automl(self):
        """Test the run_automl function for expected output type."""
        # Call run_automl with baseline mode for faster testing
        result = run_automl(
            self.temp_file.name, 
            'target', 
            task='classification',
            generations=1, 
            population_size=2,
            always_baseline=True  # Use baseline for faster testing
        )
        
        # Check that result is a dictionary with metrics
        self.assertIsInstance(result, dict)
        self.assertIn('metrics', result)
        self.assertIn('accuracy', result['metrics'])
        self.assertIsInstance(result['metrics']['accuracy'], float)

    def tearDown(self):
        """Clean up resources."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

if __name__ == '__main__':
    unittest.main()
    
