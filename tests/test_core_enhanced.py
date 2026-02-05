"""Enhanced unit tests for core functionality without requiring TPOT."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile


class TestLoadData:
    """Tests for load_data function."""

    def test_load_valid_csv(self, tmp_path):
        """Test loading a valid CSV file."""
        from core import load_data

        # Create valid CSV
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })

        csv_path = tmp_path / "test.csv"
        data.to_csv(csv_path, index=False)

        loaded = load_data(csv_path)

        assert len(loaded) == 5
        assert list(loaded.columns) == ['feature1', 'feature2', 'target']
        assert loaded['feature1'].tolist() == [1, 2, 3, 4, 5]

    def test_load_data_with_size_check(self, tmp_path):
        """Test that file size is checked."""
        from core import load_data

        data = pd.DataFrame({
            'x': range(100),
            'y': range(100)
        })

        csv_path = tmp_path / "data.csv"
        data.to_csv(csv_path, index=False)

        # Should work with large limit
        loaded = load_data(csv_path, max_size_mb=100)
        assert len(loaded) == 100

        # Should fail with tiny limit
        with pytest.raises(ValueError, match="exceeds maximum"):
            load_data(csv_path, max_size_mb=0.001)

    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        from core import load_data

        with pytest.raises(FileNotFoundError):
            load_data("/path/that/does/not/exist.csv")

    def test_load_empty_file(self, tmp_path):
        """Test loading an empty CSV."""
        from core import load_data

        csv_path = tmp_path / "empty.csv"
        pd.DataFrame().to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="empty"):
            load_data(csv_path)

    def test_load_single_column_file(self, tmp_path):
        """Test loading CSV with only one column."""
        from core import load_data

        csv_path = tmp_path / "single.csv"
        pd.DataFrame({'only_col': [1, 2, 3]}).to_csv(csv_path, index=False)

        with pytest.raises(ValueError, match="at least 2 columns"):
            load_data(csv_path)


class TestValidateDataForTraining:
    """Tests for validate_data_for_training function."""

    def test_valid_classification(self):
        """Test validation passes for valid classification data."""
        from core import validate_data_for_training

        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0, 1]
        })

        # Should not raise
        validate_data_for_training(data, 'target', 'classification')

    def test_valid_regression(self):
        """Test validation passes for valid regression data."""
        from core import validate_data_for_training

        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'target': [1.5, 2.3, 3.7, 4.1, 5.9]
        })

        # Should not raise
        validate_data_for_training(data, 'target', 'regression')

    def test_nan_in_target(self):
        """Test validation fails when target has NaN."""
        from core import validate_data_for_training

        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'target': [0, 1, np.nan, 0, 1]
        })

        with pytest.raises(ValueError, match="NaN"):
            validate_data_for_training(data, 'target', 'classification')

    def test_single_class_classification(self):
        """Test validation fails for single class."""
        from core import validate_data_for_training

        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'target': [0, 0, 0, 0, 0]
        })

        with pytest.raises(ValueError, match="at least 2 classes"):
            validate_data_for_training(data, 'target', 'classification')

    def test_class_imbalance_warning(self, caplog):
        """Test warning for severe class imbalance."""
        from core import validate_data_for_training
        import logging

        caplog.set_level(logging.WARNING)

        data = pd.DataFrame({
            'x': range(100),
            'target': [0] * 95 + [1] * 5
        })

        validate_data_for_training(data, 'target', 'classification')

        assert any('imbalance' in record.message.lower()
                   for record in caplog.records)


class TestSplitData:
    """Tests for split_data function."""

    def test_basic_split(self):
        """Test basic data splitting."""
        from core import split_data

        data = pd.DataFrame({
            'x1': range(100),
            'x2': range(100, 200),
            'target': [0, 1] * 50
        })

        X_train, X_test, y_train, y_test = split_data(
            data, 'target', test_size=0.2, random_state=42
        )

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert 'target' not in X_train.columns
        assert 'target' not in X_test.columns

    def test_stratified_split_classification(self):
        """Test stratified split for classification."""
        from core import split_data

        data = pd.DataFrame({
            'x': range(100),
            'target': [0] * 80 + [1] * 20
        })

        X_train, X_test, y_train, y_test = split_data(
            data, 'target', test_size=0.2, task='classification', random_state=42
        )

        # Check that class distribution is roughly maintained
        train_pos = (y_train == 1).sum() / len(y_train)
        test_pos = (y_test == 1).sum() / len(y_test)

        # Both should be around 0.2
        assert 0.15 <= train_pos <= 0.25
        assert 0.10 <= test_pos <= 0.30

    def test_missing_target_column(self):
        """Test error when target column missing."""
        from core import split_data

        data = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [10, 20, 30, 40, 50]
        })

        with pytest.raises(KeyError, match="not found"):
            split_data(data, 'missing_target', test_size=0.2)

    def test_insufficient_data(self):
        """Test error with too few samples."""
        from core import split_data

        data = pd.DataFrame({
            'x': [1, 2],
            'target': [0, 1]
        })

        with pytest.raises(ValueError, match="Insufficient"):
            split_data(data, 'target', test_size=0.5)


class TestMetricsComputation:
    """Tests for metrics computation functions."""

    def test_classification_metrics_basic(self):
        """Test basic classification metrics computation."""
        from core import _compute_classification_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1])

        metrics = _compute_classification_metrics(y_true, y_pred)

        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics

        # Check accuracy manually: 3 correct out of 4 = 0.75
        assert metrics['accuracy'] == 0.75

    def test_classification_metrics_with_proba(self):
        """Test classification metrics with probabilities."""
        from core import _compute_classification_metrics

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = _compute_classification_metrics(y_true, y_pred, y_proba)

        # Should include ROC AUC for binary classification
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1

    def test_regression_metrics_basic(self):
        """Test basic regression metrics computation."""
        from core import _compute_regression_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2])

        metrics = _compute_regression_metrics(y_true, y_pred)

        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics

        # RMSE should be sqrt of MSE
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']))

    def test_regression_metrics_perfect(self):
        """Test regression metrics with perfect predictions."""
        from core import _compute_regression_metrics

        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = y_true.copy()

        metrics = _compute_regression_metrics(y_true, y_pred)

        assert np.isclose(metrics['mse'], 0.0)
        assert np.isclose(metrics['rmse'], 0.0)
        assert np.isclose(metrics['mae'], 0.0)
        assert np.isclose(metrics['r2'], 1.0)


class TestLoadModel:
    """Tests for load_model function."""

    def test_load_joblib_model(self, tmp_path):
        """Test loading a model saved with joblib."""
        from core import load_model
        from sklearn.linear_model import LogisticRegression
        import joblib

        # Create and save model
        model = LogisticRegression()
        model_path = tmp_path / "model.pkl"
        joblib.dump(model, model_path)

        # Load it
        loaded = load_model(model_path)

        assert isinstance(loaded, LogisticRegression)

    def test_load_nonexistent_model(self):
        """Test loading nonexistent model."""
        from core import load_model

        with pytest.raises(FileNotFoundError):
            load_model("/path/that/does/not/exist.pkl")


class TestPredict:
    """Tests for predict function."""

    def test_predict_with_dataframe(self, tmp_path):
        """Test prediction with DataFrame input."""
        from core import predict
        from sklearn.linear_model import LogisticRegression
        import joblib

        # Create and fit a simple model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 0, 1, 1])
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Create test data
        test_data = pd.DataFrame({
            'feature1': [2, 4, 6],
            'feature2': [3, 5, 7]
        })

        # Predict
        predictions = predict(model, test_data)

        assert len(predictions) == 3
        assert all(p in [0, 1] for p in predictions)

    def test_predict_with_csv_path(self, tmp_path):
        """Test prediction with CSV file path."""
        from core import predict
        from sklearn.linear_model import LogisticRegression

        # Create and fit model
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y_train = np.array([0, 0, 1, 1])
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Create test CSV
        test_data = pd.DataFrame({
            'feature1': [2, 4, 6],
            'feature2': [3, 5, 7]
        })
        csv_path = tmp_path / "test.csv"
        test_data.to_csv(csv_path, index=False)

        # Predict
        predictions = predict(model, csv_path)

        assert len(predictions) == 3

    def test_predict_excludes_target_column(self, tmp_path):
        """Test that target column is excluded from prediction."""
        from core import predict
        from sklearn.linear_model import LogisticRegression

        # Create and fit model
        X_train = np.array([[1, 2], [3, 4]])
        y_train = np.array([0, 1])
        model = LogisticRegression()
        model.fit(X_train, y_train)

        # Create test data with target column
        test_data = pd.DataFrame({
            'feature1': [2, 4],
            'feature2': [3, 5],
            'target': [0, 1]  # Should be ignored
        })

        # Predict with target column specified
        predictions = predict(model, test_data, target_column='target')

        assert len(predictions) == 2


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_valid_config(self, tmp_path):
        """Test loading a valid YAML config."""
        from core import _load_config

        config_path = tmp_path / "config.yaml"
        config_content = """
test_size: 0.3
random_state: 123
generations: 10
"""
        config_path.write_text(config_content)

        config = _load_config(str(config_path))

        assert config['test_size'] == 0.3
        assert config['random_state'] == 123
        assert config['generations'] == 10

    def test_load_nonexistent_config(self):
        """Test loading nonexistent config returns empty dict."""
        from core import _load_config

        config = _load_config("/nonexistent/path.yaml")

        assert config == {}

    def test_load_invalid_yaml(self, tmp_path):
        """Test loading invalid YAML returns empty dict."""
        from core import _load_config

        config_path = tmp_path / "invalid.yaml"
        config_path.write_text("this is: [invalid yaml {")

        config = _load_config(str(config_path))

        assert config == {}

    def test_load_config_none_path(self):
        """Test loading config with None path."""
        from core import _load_config

        config = _load_config(None)

        assert config == {}