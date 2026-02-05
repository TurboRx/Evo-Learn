"""
Example demonstrating new v1.3.0 features:
- Data validation with helpful error messages
- File size limit protection
- Secure model serialization with joblib
- Enhanced logging and warnings

This script shows how the new validation catches common issues early.

Run from repo root: python examples/05_new_features_demo.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def example_1_valid_data():
    """Example 1: Valid data - training succeeds."""
    print("\n" + "=" * 60)
    print("Example 1: Valid Classification Data")
    print("=" * 60)

    # Create a valid classification dataset
    data = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "feature3": np.random.randn(100),
            "target": np.random.choice([0, 1], 100),
        }
    )

    # Save to CSV
    data_path = Path("/tmp/valid_data.csv")
    data.to_csv(data_path, index=False)
    print(f"✓ Created valid dataset: {data.shape}")
    print(f"✓ Class distribution: {data['target'].value_counts().to_dict()}")

    # Try to train - should succeed
    from core import run_automl

    try:
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            generations=2,  # Quick test
            population_size=10,
            always_baseline=True,  # Use baseline for speed
        )
        print(f"✓ Training successful! Accuracy: {result['metrics']['accuracy']:.4f}")
    except Exception as e:
        print(f"✗ Training failed: {e}")

    # Cleanup
    data_path.unlink(missing_ok=True)


def example_2_nan_in_target():
    """Example 2: NaN in target - validation catches it."""
    print("\n" + "=" * 60)
    print("Example 2: NaN in Target Column (Should Fail Validation)")
    print("=" * 60)

    # Create dataset with NaN in target
    data = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": [0, 1, np.nan, 0, 1] * 20,  # Contains NaN
        }
    )

    data_path = Path("/tmp/nan_target.csv")
    data.to_csv(data_path, index=False)
    print(f"✓ Created dataset with NaN in target: {data['target'].isna().sum()} NaNs")

    # Try to train - should fail with clear error
    from core import run_automl

    try:
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            generations=2,
            population_size=10,
            always_baseline=True,
        )
        print("✗ Training should have failed but didn't!")
    except ValueError as e:
        print(f"✓ Validation correctly caught the issue:")
        print(f"  Error: {str(e)[:100]}...")

    # Cleanup
    data_path.unlink(missing_ok=True)


def example_3_single_class():
    """Example 3: Single class - validation catches it."""
    print("\n" + "=" * 60)
    print("Example 3: Single Class in Classification (Should Fail)")
    print("=" * 60)

    # Create dataset with only one class
    data = pd.DataFrame(
        {
            "feature1": np.random.randn(100),
            "feature2": np.random.randn(100),
            "target": [0] * 100,  # All same class
        }
    )

    data_path = Path("/tmp/single_class.csv")
    data.to_csv(data_path, index=False)
    print(f"✓ Created dataset with single class: {data['target'].unique()}")

    # Try to train - should fail with clear error
    from core import run_automl

    try:
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            generations=2,
            population_size=10,
            always_baseline=True,
        )
        print("✗ Training should have failed but didn't!")
    except ValueError as e:
        print(f"✓ Validation correctly caught the issue:")
        print(f"  Error: {str(e)[:100]}...")

    # Cleanup
    data_path.unlink(missing_ok=True)


def example_4_class_imbalance_warning():
    """Example 4: Class imbalance - validation warns but continues."""
    print("\n" + "=" * 60)
    print("Example 4: Severe Class Imbalance (Warning Only)")
    print("=" * 60)

    # Create imbalanced dataset (20:1 ratio)
    data = pd.DataFrame(
        {
            "feature1": np.random.randn(200),
            "feature2": np.random.randn(200),
            "target": [0] * 190 + [1] * 10,  # 19:1 ratio
        }
    )

    data_path = Path("/tmp/imbalanced.csv")
    data.to_csv(data_path, index=False)
    print(f"✓ Created imbalanced dataset:")
    print(f"  Class 0: {(data['target'] == 0).sum()} samples")
    print(f"  Class 1: {(data['target'] == 1).sum()} samples")

    # Try to train - should warn but succeed
    from core import run_automl

    try:
        print("\n  Training with imbalanced data...")
        result = run_automl(
            data_path=data_path,
            target_column="target",
            task="classification",
            generations=2,
            population_size=10,
            always_baseline=True,
        )
        print(
            f"✓ Training succeeded with warning. Accuracy: {result['metrics']['accuracy']:.4f}"
        )
    except Exception as e:
        print(f"✗ Training failed: {e}")

    # Cleanup
    data_path.unlink(missing_ok=True)


def example_5_model_serialization():
    """Example 5: Demonstrate secure model serialization with joblib."""
    print("\n" + "=" * 60)
    print("Example 5: Secure Model Serialization (joblib)")
    print("=" * 60)

    from sklearn.linear_model import LogisticRegression
    import joblib

    # Create and save a model
    model = LogisticRegression(max_iter=100)
    model.coef_ = np.array([[1.0, 2.0, 3.0]])
    model.intercept_ = np.array([0.5])
    model.classes_ = np.array([0, 1])

    model_path = Path("/tmp/test_model.pkl")
    print("✓ Saving model with joblib (secure)...")
    joblib.dump(model, model_path, compress=3)
    print(f"  Saved to: {model_path}")
    print(f"  File size: {model_path.stat().st_size} bytes")

    # Load the model
    print("\n✓ Loading model with joblib...")
    loaded_model = joblib.load(model_path)
    print(f"  Model type: {type(loaded_model).__name__}")
    print(f"  Coefficients match: {np.allclose(model.coef_, loaded_model.coef_)}")

    # Cleanup
    model_path.unlink(missing_ok=True)


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Evo-Learn v1.3.0 New Features Demo")
    print("=" * 60)

    # Run examples
    example_1_valid_data()
    example_2_nan_in_target()
    example_3_single_class()
    example_4_class_imbalance_warning()
    example_5_model_serialization()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print(
        """
Key Takeaways:
1. Data validation catches issues BEFORE training starts
2. Clear error messages help you fix problems quickly
3. Warnings inform you about potential issues (but training continues)
4. Joblib provides secure, compressed model serialization
5. All improvements maintain backward compatibility
"""
    )
