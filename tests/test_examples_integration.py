"""Integration tests for the examples/05_new_features_demo.py."""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import run_automl, load_data, validate_data_for_training


class TestNewFeaturesDemo:
    """Integration tests for new v1.3.0 features demonstration."""

    def test_valid_data_training_succeeds(self, tmp_path):
        """Test Example 1: Valid data - training succeeds."""
        # Create a valid classification dataset
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })

        # Save to CSV
        data_path = tmp_path / "valid_data.csv"
        data.to_csv(data_path, index=False)

        # Try to train - should succeed
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            generations=2,
            population_size=10,
            always_baseline=True,
            output_dir=tmp_path / "models"
        )

        # Verify success
        assert result is not None
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
        assert 0.0 <= result['metrics']['accuracy'] <= 1.0

    def test_nan_in_target_validation_error(self, tmp_path):
        """Test Example 2: NaN in target - validation catches it."""
        # Create dataset with NaN in target
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': [0, 1, np.nan, 0, 1] * 20
        })

        data_path = tmp_path / "nan_target.csv"
        data.to_csv(data_path, index=False)

        # Try to train - should fail with clear error
        with pytest.raises(ValueError, match="NaN"):
            run_automl(
                data_path=data_path,
                target_column="target",
                task="classification",
                generations=2,
                population_size=10,
                always_baseline=True,
                output_dir=tmp_path / "models"
            )

    def test_single_class_validation_error(self, tmp_path):
        """Test Example 3: Single class - validation catches it."""
        # Create dataset with only one class
        data = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': [0] * 100
        })

        data_path = tmp_path / "single_class.csv"
        data.to_csv(data_path, index=False)

        # Try to train - should fail with clear error
        with pytest.raises(ValueError, match="at least 2 classes"):
            run_automl(
                data_path=data_path,
                target_column="target",
                task="classification",
                generations=2,
                population_size=10,
                always_baseline=True,
                output_dir=tmp_path / "models"
            )

    def test_class_imbalance_warning_continues(self, tmp_path, caplog):
        """Test Example 4: Class imbalance - validation warns but continues."""
        import logging
        caplog.set_level(logging.WARNING)

        # Create imbalanced dataset (20:1 ratio)
        data = pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'target': [0] * 190 + [1] * 10
        })

        data_path = tmp_path / "imbalanced.csv"
        data.to_csv(data_path, index=False)

        # Try to train - should warn but succeed
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            generations=2,
            population_size=10,
            always_baseline=True,
            output_dir=tmp_path / "models"
        )

        # Should succeed with warning
        assert result is not None
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']

        # Check that warning was logged
        assert any('imbalance' in record.message.lower()
                   for record in caplog.records)

    def test_model_serialization_with_joblib(self, tmp_path):
        """Test Example 5: Secure model serialization with joblib."""
        from sklearn.linear_model import LogisticRegression
        import joblib

        # Create and save a model
        model = LogisticRegression(max_iter=100)
        model.coef_ = np.array([[1.0, 2.0, 3.0]])
        model.intercept_ = np.array([0.5])
        model.classes_ = np.array([0, 1])

        model_path = tmp_path / "test_model.pkl"
        joblib.dump(model, model_path, compress=3)

        # Verify file was created
        assert model_path.exists()
        assert model_path.stat().st_size > 0

        # Load the model
        loaded_model = joblib.load(model_path)

        # Verify it's correct
        assert type(loaded_model).__name__ == 'LogisticRegression'
        assert np.allclose(model.coef_, loaded_model.coef_)
        assert np.allclose(model.intercept_, loaded_model.intercept_)


class TestEndToEndWorkflow:
    """End-to-end integration tests for complete workflows."""

    def test_complete_classification_workflow(self, tmp_path):
        """Test complete workflow: load, validate, train, save, load, predict."""
        # Step 1: Create and save data
        np.random.seed(42)
        data = pd.DataFrame({
            'numeric1': np.random.randn(100),
            'numeric2': np.random.randn(100),
            'categorical': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })

        data_path = tmp_path / "full_workflow.csv"
        data.to_csv(data_path, index=False)

        # Step 2: Load data
        loaded_data = load_data(data_path)
        assert len(loaded_data) == 100

        # Step 3: Validate data
        validate_data_for_training(loaded_data, 'target', 'classification')

        # Step 4: Train model
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            generations=2,
            population_size=10,
            always_baseline=True,
            output_dir=tmp_path / "models"
        )

        assert result is not None
        assert 'model_path' in result
        model_path = result['model_path']

        # Step 5: Load trained model
        from core import load_model
        model = load_model(model_path)
        assert model is not None

        # Step 6: Make predictions
        from core import predict
        predictions = predict(model, data_path, target_column='target')
        assert len(predictions) == 100
        assert all(p in [0, 1] for p in predictions)

    def test_complete_regression_workflow(self, tmp_path):
        """Test complete workflow for regression."""
        # Create regression data
        np.random.seed(42)
        X1 = np.random.randn(100)
        X2 = np.random.randn(100)
        y = 2 * X1 + 3 * X2 + np.random.randn(100) * 0.1

        data = pd.DataFrame({
            'feature1': X1,
            'feature2': X2,
            'target': y
        })

        data_path = tmp_path / "regression_data.csv"
        data.to_csv(data_path, index=False)

        # Train model
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="regression",
            generations=2,
            population_size=10,
            always_baseline=True,
            output_dir=tmp_path / "models"
        )

        # Verify regression metrics
        assert result is not None
        assert 'metrics' in result
        assert 'mse' in result['metrics']
        assert 'rmse' in result['metrics']
        assert 'mae' in result['metrics']
        assert 'r2' in result['metrics']

    def test_workflow_with_missing_values(self, tmp_path):
        """Test workflow with missing values in features (not target)."""
        # Create data with missing values in features
        data = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 6, 7, 8],
            'feature2': [10, np.nan, 30, 40, 50, np.nan, 70, 80],
            'feature3': [100, 200, 300, 400, 500, 600, 700, 800],
            'target': [0, 1, 0, 1, 0, 1, 0, 1]
        })

        data_path = tmp_path / "missing_values.csv"
        data.to_csv(data_path, index=False)

        # Should handle missing values in features gracefully
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            generations=2,
            population_size=10,
            always_baseline=True,
            output_dir=tmp_path / "models"
        )

        assert result is not None
        assert 'metrics' in result

    def test_workflow_with_config_file(self, tmp_path):
        """Test workflow using configuration file."""
        # Create data
        data = pd.DataFrame({
            'feature1': np.random.randn(50),
            'feature2': np.random.randn(50),
            'target': [0, 1] * 25
        })

        data_path = tmp_path / "config_test.csv"
        data.to_csv(data_path, index=False)

        # Create config file
        config_path = tmp_path / "test_config.yaml"
        config_content = """
default_task: classification
random_state: 42
test_size: 0.2
generations: 2
population_size: 10
handle_categoricals: true
impute_strategy: median
scale_numeric: true
baseline: true
output_dir: models
"""
        config_path.write_text(config_content)

        # Train with config
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            config_path=str(config_path),
            output_dir=tmp_path / "models"
        )

        assert result is not None
        assert 'metrics' in result

    def test_baseline_vs_tpot_fallback(self, tmp_path):
        """Test that baseline fallback works when TPOT might fail."""
        # Create minimal dataset that might cause TPOT issues
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'target': [0, 1, 0, 1, 0, 1]
        })

        data_path = tmp_path / "minimal.csv"
        data.to_csv(data_path, index=False)

        # Use baseline explicitly
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            always_baseline=True,
            output_dir=tmp_path / "models"
        )

        assert result is not None
        assert result['model_type'] in ['baseline_logreg', 'baseline_ridge']


class TestDataValidationIntegration:
    """Integration tests for data validation in real workflows."""

    def test_validation_catches_issues_before_expensive_training(self, tmp_path):
        """Test that validation catches issues early, before TPOT starts."""
        # Create invalid data (single class)
        data = pd.DataFrame({
            'feature1': np.random.randn(1000),
            'feature2': np.random.randn(1000),
            'target': [0] * 1000
        })

        data_path = tmp_path / "invalid_large.csv"
        data.to_csv(data_path, index=False)

        # Should fail fast without waiting for TPOT
        import time
        start_time = time.time()

        with pytest.raises(ValueError, match="at least 2 classes"):
            run_automl(
                data_path=data_path,
                target_column="target",
                task="classification",
                output_dir=tmp_path / "models"
            )

        elapsed_time = time.time() - start_time

        # Should fail quickly (within 5 seconds), not after TPOT runs
        assert elapsed_time < 5.0

    def test_validation_allows_valid_edge_cases(self, tmp_path):
        """Test that validation allows valid edge cases."""
        # Small but valid dataset
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })

        data_path = tmp_path / "small_valid.csv"
        data.to_csv(data_path, index=False)

        # Should succeed
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            always_baseline=True,
            output_dir=tmp_path / "models"
        )

        assert result is not None


class TestBackwardCompatibility:
    """Tests to ensure backward compatibility with existing code."""

    def test_old_pickle_models_still_load(self, tmp_path):
        """Test that models saved with old pickle format can still be loaded."""
        from sklearn.linear_model import LogisticRegression
        import pickle
        from core import load_model

        # Create and save with old pickle format
        model = LogisticRegression()
        model.coef_ = np.array([[1.0, 2.0]])
        model.intercept_ = np.array([0.5])
        model.classes_ = np.array([0, 1])

        model_path = tmp_path / "old_pickle_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Should still load with load_model (which uses joblib)
        loaded_model = load_model(model_path)
        assert isinstance(loaded_model, LogisticRegression)

    def test_api_unchanged_for_basic_usage(self, tmp_path):
        """Test that basic API usage patterns remain unchanged."""
        # Create simple dataset
        data = pd.DataFrame({
            'x': np.random.randn(50),
            'y': [0, 1] * 25
        })

        data_path = tmp_path / "api_test.csv"
        data.to_csv(data_path, index=False)

        # Old-style API call should still work
        result = run_automl(
            data_path,
            'y',
            task='classification',
            always_baseline=True
        )

        # Should return expected structure
        assert isinstance(result, dict)
        assert 'metrics' in result
        assert 'model_path' in result