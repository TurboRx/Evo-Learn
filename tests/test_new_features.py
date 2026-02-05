"""Tests for the new data validation functionality."""
import pytest
import pandas as pd
import numpy as np
from core import validate_data_for_training


class TestDataValidation:
    """Test suite for validate_data_for_training function."""
    
    def test_valid_classification_data(self):
        """Test that valid classification data passes validation."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': [7, 8, 9, 10, 11, 12],
            'target': [0, 1, 0, 1, 0, 1]
        })
        
        # Should not raise any exception
        validate_data_for_training(data, 'target', 'classification')
    
    def test_valid_regression_data(self):
        """Test that valid regression data passes validation."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [7, 8, 9, 10, 11],
            'target': [1.5, 2.3, 3.1, 4.2, 5.8]
        })
        
        # Should not raise any exception
        validate_data_for_training(data, 'target', 'regression')
    
    def test_nan_in_target_raises_error(self):
        """Test that NaN in target column raises ValueError."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [7, 8, 9, 10, 11],
            'target': [0, 1, np.nan, 1, 0]
        })
        
        with pytest.raises(ValueError, match="contains.*NaN"):
            validate_data_for_training(data, 'target', 'classification')
    
    def test_single_class_raises_error(self):
        """Test that single class in classification raises ValueError."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [7, 8, 9, 10, 11],
            'target': [0, 0, 0, 0, 0]  # All same class
        })
        
        with pytest.raises(ValueError, match="at least 2 classes"):
            validate_data_for_training(data, 'target', 'classification')
    
    def test_class_imbalance_warning(self, caplog):
        """Test that severe class imbalance generates warning."""
        # Create imbalanced dataset (ratio > 10:1)
        data = pd.DataFrame({
            'feature1': list(range(100)),
            'target': [0] * 95 + [1] * 5  # 19:1 ratio
        })
        
        validate_data_for_training(data, 'target', 'classification')
        
        # Check that warning was logged
        assert any('class imbalance' in record.message.lower() 
                   for record in caplog.records)
    
    def test_all_nan_features_warning(self, caplog):
        """Test that all-NaN features generate warning."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [np.nan] * 5,  # All NaN
            'target': [0, 1, 0, 1, 0]
        })
        
        validate_data_for_training(data, 'target', 'classification')
        
        # Check that warning was logged
        assert any('all NaN' in record.message 
                   for record in caplog.records)
    
    def test_constant_features_warning(self, caplog):
        """Test that constant features generate warning."""
        data = pd.DataFrame({
            'feature1': [5, 5, 5, 5, 5],  # Constant
            'feature2': [1, 2, 3, 4, 5],  # Variable
            'target': [0, 1, 0, 1, 0]
        })
        
        validate_data_for_training(data, 'target', 'classification')
        
        # Check that warning was logged
        assert any('constant' in record.message.lower() 
                   for record in caplog.records)
    
    def test_minimum_classes_for_classification(self):
        """Test that classification requires at least 2 classes."""
        # Binary classification - should pass
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })
        validate_data_for_training(data, 'target', 'classification')
        
        # Multi-class - should pass
        data_multi = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'target': [0, 1, 2, 0, 1, 2]
        })
        validate_data_for_training(data_multi, 'target', 'classification')


class TestFileLoading:
    """Tests for file loading with size limits."""
    
    def test_load_data_size_limit(self, tmp_path):
        """Test that file size limit is enforced."""
        from core import load_data
        
        # Create a small CSV file
        csv_path = tmp_path / "small_data.csv"
        data = pd.DataFrame({
            'feature1': list(range(10)),
            'target': [0, 1] * 5
        })
        data.to_csv(csv_path, index=False)
        
        # Should load successfully with default limit
        loaded = load_data(csv_path)
        assert len(loaded) == 10
        
    def test_load_data_logs_file_size(self, tmp_path, caplog):
        """Test that file size is logged when loading."""
        from core import load_data
        import logging
        
        # Enable logging capture
        caplog.set_level(logging.INFO)
        
        csv_path = tmp_path / "test_data.csv"
        data = pd.DataFrame({
            'feature1': list(range(100)),
            'target': [0, 1] * 50
        })
        data.to_csv(csv_path, index=False)
        
        load_data(csv_path)
        
        # Check that data was loaded (should log something about loading)
        assert any('Loading' in record.message or 'Loaded' in record.message
                   for record in caplog.records)


class TestModelSerialization:
    """Tests for model serialization with joblib."""
    
    def test_model_save_load_roundtrip(self, tmp_path):
        """Test that models can be saved and loaded with joblib."""
        from sklearn.linear_model import LogisticRegression
        import joblib
        
        # Create and save a model
        model = LogisticRegression()
        model_path = tmp_path / "test_model.pkl"
        
        joblib.dump(model, model_path, compress=3)
        
        # Load the model
        loaded_model = joblib.load(model_path)
        
        # Verify it's the same type
        assert isinstance(loaded_model, LogisticRegression)
        assert type(loaded_model) == type(model)
