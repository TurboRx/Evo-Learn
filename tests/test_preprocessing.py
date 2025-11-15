# tests/test_preprocessing.py

import unittest
import pandas as pd
import tempfile
import os
import numpy as np
from preprocessing import build_preprocessor

class TestPreprocessing(unittest.TestCase):
    """Unit tests for data preprocessing functionalities."""

    def setUp(self):
        """Set up a mock dataset in a temporary file."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'category': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })
        
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        data.to_csv(self.temp_file.name, index=False)
        self.temp_file.close()

    def test_preprocess_data(self):
        """
        Test build_preprocessor function for:
        - Handling categorical features (encoding).
        - Handling numeric features (imputation and scaling).
        """
        # Load the data
        data = pd.read_csv(self.temp_file.name)
        
        # Build preprocessor
        preprocessor, feature_names = build_preprocessor(
            data, 
            target_column='target',
            handle_categoricals=True,
            scale_numeric=True
        )
        
        # Test that preprocessor was created
        self.assertIsNotNone(preprocessor)
        
        # Transform the data
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)
        
        # Check that output is numeric
        self.assertTrue(np.issubdtype(X_transformed.dtype, np.number))
        
        # Check that we have the expected number of features
        # feature1 (1) + category one-hot encoded (2) = 3 features
        self.assertEqual(X_transformed.shape[1], 3)

    def tearDown(self):
        """Clean up resources."""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)

if __name__ == '__main__':
    unittest.main()
    
