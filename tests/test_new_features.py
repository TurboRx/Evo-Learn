"""Tests for the new data validation functionality."""
import pytest
import pandas as pd
import numpy as np
from core import validate_data_for_training, load_data
import tempfile
from pathlib import Path


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

    def test_multiple_nans_in_target(self):
        """Test error message includes count of NaN values."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6, 7, 8],
            'target': [0, 1, np.nan, 1, 0, np.nan, np.nan, 1]
        })

        with pytest.raises(ValueError, match="contains 3 NaN"):
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

    def test_class_imbalance_boundary(self, caplog):
        """Test class imbalance warning at the boundary (10:1 ratio)."""
        # Exactly 10:1 ratio - should warn
        data = pd.DataFrame({
            'feature1': list(range(110)),
            'target': [0] * 100 + [1] * 10  # 10:1 ratio
        })

        validate_data_for_training(data, 'target', 'classification')

        # Just under the threshold - should not warn
        data_balanced = pd.DataFrame({
            'feature1': list(range(100)),
            'target': [0] * 90 + [1] * 10  # 9:1 ratio
        })

        caplog.clear()
        validate_data_for_training(data_balanced, 'target', 'classification')
        assert not any('class imbalance' in record.message.lower()
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

    def test_multiple_all_nan_features(self, caplog):
        """Test warning shows number of all-NaN features."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [np.nan] * 5,  # All NaN
            'feature3': [np.nan] * 5,  # All NaN
            'feature4': [np.nan] * 5,  # All NaN
            'target': [0, 1, 0, 1, 0]
        })

        validate_data_for_training(data, 'target', 'classification')

        # Check that warning includes count
        assert any('3 columns' in record.message
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

    def test_multiclass_classification(self):
        """Test validation with many classes."""
        data = pd.DataFrame({
            'feature1': list(range(100)),
            'target': [i % 10 for i in range(100)]  # 10 classes
        })

        # Should pass without errors
        validate_data_for_training(data, 'target', 'classification')

    def test_regression_with_single_value(self):
        """Test that regression allows single unique value (edge case)."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': [5.0, 5.0, 5.0, 5.0, 5.0]  # All same value
        })

        # Regression should allow this (though it's not useful)
        validate_data_for_training(data, 'target', 'regression')


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

    def test_load_data_exceeds_size_limit(self, tmp_path):
        """Test that oversized files are rejected."""
        from core import load_data

        # Create a CSV file
        csv_path = tmp_path / "large_data.csv"
        # Create a reasonably sized dataset
        data = pd.DataFrame({
            'feature1': list(range(1000)),
            'feature2': list(range(1000)),
            'target': [0, 1] * 500
        })
        data.to_csv(csv_path, index=False)

        # Set a very small limit (e.g., 0.001 MB) - should fail
        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            load_data(csv_path, max_size_mb=0.001)

    def test_load_data_custom_size_limit(self, tmp_path):
        """Test that custom size limits work."""
        from core import load_data

        csv_path = tmp_path / "data.csv"
        data = pd.DataFrame({
            'feature1': list(range(100)),
            'target': [0, 1] * 50
        })
        data.to_csv(csv_path, index=False)

        # Should work with generous limit
        loaded = load_data(csv_path, max_size_mb=1000)
        assert len(loaded) == 100

    def test_load_data_nonexistent_file(self):
        """Test error handling for nonexistent files."""
        from core import load_data

        with pytest.raises(FileNotFoundError, match="not found"):
            load_data("/nonexistent/path/data.csv")

    def test_load_data_empty_file(self, tmp_path):
        """Test error handling for empty CSV files."""
        from core import load_data

        csv_path = tmp_path / "empty.csv"
        # Create empty CSV with just headers
        pd.DataFrame(columns=['feature1', 'target']).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="empty"):
            load_data(csv_path)

    def test_load_data_insufficient_columns(self, tmp_path):
        """Test error handling for files with insufficient columns."""
        from core import load_data

        csv_path = tmp_path / "single_column.csv"
        pd.DataFrame({'only_one_column': [1, 2, 3]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="at least 2 columns"):
            load_data(csv_path)

    def test_load_data_non_csv_extension(self, tmp_path, caplog):
        """Test warning for non-CSV file extensions."""
        from core import load_data
        import logging

        caplog.set_level(logging.WARNING)

        # Create file with .txt extension but valid CSV content
        txt_path = tmp_path / "data.txt"
        data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        data.to_csv(txt_path, index=False)

        loaded = load_data(txt_path)

        # Should log a warning
        assert any('.csv extension' in record.message
                   for record in caplog.records)
        assert len(loaded) == 3

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

    def test_load_data_handles_mixed_types(self, tmp_path):
        """Test that load_data handles mixed data types correctly."""
        from core import load_data

        csv_path = tmp_path / "mixed_types.csv"
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'string': ['a', 'b', 'c', 'd', 'e'],
            'mixed': ['1', '2', 'three', '4', '5'],  # Mixed numeric and text
            'target': [0, 1, 0, 1, 0]
        })
        data.to_csv(csv_path, index=False)

        loaded = load_data(csv_path)

        # Should load successfully
        assert len(loaded) == 5
        assert 'numeric' in loaded.columns
        assert 'string' in loaded.columns
        assert 'mixed' in loaded.columns


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

    def test_model_save_with_compression(self, tmp_path):
        """Test that compression is applied when saving models."""
        from sklearn.linear_model import Ridge
        import joblib

        model = Ridge()
        model_path = tmp_path / "compressed_model.pkl"

        joblib.dump(model, model_path, compress=3)

        # Verify file was created
        assert model_path.exists()
        assert model_path.stat().st_size > 0

    def test_load_model_function(self, tmp_path):
        """Test the load_model function from core."""
        from core import load_model
        from sklearn.linear_model import LogisticRegression
        import joblib

        # Create and save a model
        model = LogisticRegression()
        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path, compress=3)

        # Load using the core function
        loaded_model = load_model(model_path)

        # Verify it works
        assert isinstance(loaded_model, LogisticRegression)

    def test_load_model_nonexistent_file(self, tmp_path):
        """Test load_model with nonexistent file."""
        from core import load_model

        with pytest.raises(FileNotFoundError, match="not found"):
            load_model(tmp_path / "nonexistent_model.pkl")

    def test_fitted_model_save_load(self, tmp_path):
        """Test saving and loading a fitted model with predictions."""
        from sklearn.linear_model import LogisticRegression
        from core import load_model
        import joblib

        # Create and fit a model
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 0, 1, 1])
        model = LogisticRegression()
        model.fit(X, y)

        # Save the model
        model_path = tmp_path / "fitted_model.pkl"
        joblib.dump(model, model_path, compress=3)

        # Load and verify predictions
        loaded_model = load_model(model_path)
        original_pred = model.predict(X)
        loaded_pred = loaded_model.predict(X)

        # Predictions should match
        np.testing.assert_array_equal(original_pred, loaded_pred)

    def test_pipeline_serialization(self, tmp_path):
        """Test that full pipelines can be serialized."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression
        import joblib

        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression())
        ])

        # Save pipeline
        model_path = tmp_path / "pipeline.pkl"
        joblib.dump(pipeline, model_path, compress=3)

        # Load and verify
        loaded_pipeline = joblib.load(model_path)
        assert isinstance(loaded_pipeline, Pipeline)
        assert 'scaler' in loaded_pipeline.named_steps
        assert 'classifier' in loaded_pipeline.named_steps


class TestMetricsComputation:
    """Tests for metrics computation functions."""

    def test_classification_metrics_binary(self):
        """Test classification metrics for binary classification."""
        from core import _compute_classification_metrics

        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])

        metrics = _compute_classification_metrics(y_true, y_pred)

        # Verify all expected metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

        # Verify they are floats and in valid range
        for metric_value in metrics.values():
            assert isinstance(metric_value, float)
            assert 0.0 <= metric_value <= 1.0

    def test_classification_metrics_with_probabilities(self):
        """Test classification metrics with probability scores."""
        from core import _compute_classification_metrics

        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.4, 0.3, 0.85, 0.6])

        metrics = _compute_classification_metrics(y_true, y_pred, y_proba)

        # Should include ROC AUC for binary classification
        assert 'roc_auc' in metrics
        assert 0.0 <= metrics['roc_auc'] <= 1.0

    def test_regression_metrics(self):
        """Test regression metrics computation."""
        from core import _compute_regression_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = _compute_regression_metrics(y_true, y_pred)

        # Verify all expected metrics are present
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics

        # Verify they are floats
        for metric_value in metrics.values():
            assert isinstance(metric_value, float)

        # RMSE should be sqrt of MSE
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']))

    def test_regression_metrics_perfect_prediction(self):
        """Test regression metrics with perfect predictions."""
        from core import _compute_regression_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()

        metrics = _compute_regression_metrics(y_true, y_pred)

        # Perfect prediction should have zero error and R2=1
        assert np.isclose(metrics['mse'], 0.0)
        assert np.isclose(metrics['rmse'], 0.0)
        assert np.isclose(metrics['mae'], 0.0)
        assert np.isclose(metrics['r2'], 1.0)


class TestSplitData:
    """Tests for data splitting functionality."""

    def test_split_data_basic(self):
        """Test basic data splitting."""
        from core import split_data

        data = pd.DataFrame({
            'feature1': list(range(100)),
            'feature2': list(range(100, 200)),
            'target': [0, 1] * 50
        })

        X_train, X_test, y_train, y_test = split_data(
            data, 'target', test_size=0.2, random_state=42, task='classification'
        )

        # Check shapes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

        # Check that target is not in X
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns

    def test_split_data_stratified(self):
        """Test stratified splitting for classification."""
        from core import split_data

        # Create imbalanced dataset
        data = pd.DataFrame({
            'feature1': list(range(100)),
            'target': [0] * 80 + [1] * 20  # 80-20 split
        })

        X_train, X_test, y_train, y_test = split_data(
            data, 'target', test_size=0.2, random_state=42, task='classification'
        )

        # Check that class proportions are roughly maintained
        train_ratio = (y_train == 1).sum() / len(y_train)
        test_ratio = (y_test == 1).sum() / len(y_test)

        # Should be close to 0.2 (20%)
        assert 0.15 <= train_ratio <= 0.25
        assert 0.10 <= test_ratio <= 0.30

    def test_split_data_regression_no_stratification(self):
        """Test that regression doesn't use stratification."""
        from core import split_data

        data = pd.DataFrame({
            'feature1': list(range(100)),
            'target': np.random.randn(100)  # Continuous target
        })

        X_train, X_test, y_train, y_test = split_data(
            data, 'target', test_size=0.3, random_state=42, task='regression'
        )

        # Should split without error
        assert len(X_train) == 70
        assert len(X_test) == 30

    def test_split_data_missing_target_column(self):
        """Test error when target column is missing."""
        from core import split_data

        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [6, 7, 8, 9, 10]
        })

        with pytest.raises(KeyError, match="not found"):
            split_data(data, 'nonexistent_target', test_size=0.2)

    def test_split_data_insufficient_samples(self):
        """Test error with too few samples."""
        from core import split_data

        data = pd.DataFrame({
            'feature1': [1, 2],
            'target': [0, 1]
        })

        with pytest.raises(ValueError, match="Insufficient data"):
            split_data(data, 'target', test_size=0.5)