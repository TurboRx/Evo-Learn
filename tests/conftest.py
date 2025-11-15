"""Pytest configuration and shared fixtures."""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_classification_data():
    """Generate sample classification dataset."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature4': np.random.randn(n_samples),
    })
    
    # Add some missing values
    X.loc[X.sample(frac=0.1).index, 'feature1'] = np.nan
    X.loc[X.sample(frac=0.05).index, 'feature2'] = np.nan
    
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.6, 0.4]))
    
    return X, y


@pytest.fixture
def sample_regression_data():
    """Generate sample regression dataset."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.choice(['X', 'Y', 'Z'], n_samples),
        'feature4': np.random.randn(n_samples),
    })
    
    # Target with some noise
    y = pd.Series(
        2 * X['feature1'] + 3 * X['feature2'] + np.random.randn(n_samples) * 0.5
    )
    
    return X, y


@pytest.fixture
def sample_csv_file(tmp_path, sample_classification_data):
    """Create a temporary CSV file with sample data."""
    X, y = sample_classification_data
    data = X.copy()
    data['target'] = y
    
    csv_path = tmp_path / "sample_data.csv"
    data.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory for test artifacts."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    yield output_dir
    # Cleanup handled automatically by tmp_path


@pytest.fixture
def sample_config(temp_output_dir):
    """Generate sample configuration dictionary."""
    return {
        'default_task': 'classification',
        'random_state': 42,
        'test_size': 0.2,
        'output_dir': str(temp_output_dir),
        'generations': 2,
        'population_size': 10,
        'max_time_mins': 1,
        'max_eval_time_mins': 0.5,
        'handle_categoricals': True,
        'impute_strategy': 'median',
        'scale_numeric': True,
        'baseline': False,
    }


@pytest.fixture
def imbalanced_classification_data():
    """Generate imbalanced classification dataset."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
    })
    
    # Heavily imbalanced: 90% class 0, 10% class 1
    y = pd.Series(np.random.choice([0, 1], n_samples, p=[0.9, 0.1]))
    
    return X, y


@pytest.fixture
def multiclass_data():
    """Generate multiclass classification dataset."""
    np.random.seed(42)
    n_samples = 300
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
    })
    
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    
    return X, y
