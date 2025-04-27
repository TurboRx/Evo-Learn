# tests/test_preprocessing.py

import unittest
import pandas as pd
from mloptimizer.preprocessing import preprocess_data

class TestPreprocessing(unittest.TestCase):

    def test_preprocess_data(self):
        # Create a simple test dataset with categorical features
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'category': ['A', 'B', 'A']
        })
        data.to_csv('test_data.csv', index=False)

        # Preprocess the data and check if categorical features are encoded
        preprocessed_data = preprocess_data('test_data.csv', categorical_features=['category'])
        self.assertTrue(all(isinstance(x, (int, float)) for x in preprocessed_data['category']))

if __name__ == '__main__':
    unittest.main()
    
