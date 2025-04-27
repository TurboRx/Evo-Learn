# tests/test_preprocessing.py

import unittest
import pandas as pd
from io import StringIO
from mloptimizer.preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):
    """Unit tests for data preprocessing functionalities."""

    def setUp(self):
        """Set up a mock dataset in memory."""
        self.mock_data = StringIO("""
        feature1,category
        1,A
        2,B
        3,A
        """)

    def test_preprocess_data(self):
        """
        Test preprocess_data function for:
        - Encoding categorical features (e.g., columns with string values).
        """
        # Load the mock data into a DataFrame
        data = pd.read_csv(self.mock_data)

        # Preprocess the data and validate results
        preprocessed_data = preprocess_data(self.mock_data, categorical_features=['category'])
        self.assertTrue(all(isinstance(x, (int, float)) for x in preprocessed_data['category']))

    def tearDown(self):
        """Clean up resources."""
        self.mock_data.close()

if __name__ == '__main__':
    unittest.main()
    
