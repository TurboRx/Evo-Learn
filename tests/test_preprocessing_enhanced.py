"""Enhanced tests for preprocessing functionality."""
import pytest
import pandas as pd
import numpy as np
from preprocessing import build_preprocessor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


class TestBuildPreprocessor:
    """Comprehensive tests for build_preprocessor function."""

    def test_numeric_only_with_scaling(self):
        """Test preprocessor with only numeric features and scaling."""
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, feature_names = build_preprocessor(
            data, 'target', scale_numeric=True, handle_categoricals=True
        )

        # Should have numeric transformer
        assert 'num' in preprocessor.named_transformers_

        # Transform and check shape
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)
        assert X_transformed.shape[1] == 2  # 2 numeric features

    def test_numeric_only_without_scaling(self):
        """Test preprocessor with numeric features but no scaling."""
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, feature_names = build_preprocessor(
            data, 'target', scale_numeric=False, handle_categoricals=True
        )

        # Should still have numeric transformer (for imputation)
        assert 'num' in preprocessor.named_transformers_

        # Transform data
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)
        assert X_transformed.shape[1] == 2

    def test_categorical_only(self):
        """Test preprocessor with only categorical features."""
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X'],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, feature_names = build_preprocessor(
            data, 'target', handle_categoricals=True
        )

        # Should have categorical transformer
        assert 'cat' in preprocessor.named_transformers_

        # Transform and check shape
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)
        # cat1: 2 categories, cat2: 2 categories = 4 features
        assert X_transformed.shape[1] == 4

    def test_mixed_numeric_categorical(self):
        """Test preprocessor with both numeric and categorical features."""
        data = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric2': [10, 20, 30, 40, 50],
            'cat1': ['A', 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'Y', 'Z', 'X', 'Y'],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, feature_names = build_preprocessor(
            data, 'target', handle_categoricals=True, scale_numeric=True
        )

        # Should have both transformers
        assert 'num' in preprocessor.named_transformers_
        assert 'cat' in preprocessor.named_transformers_

        # Transform and check shape
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)
        # 2 numeric + (2 cat1 + 3 cat2) = 7 features
        assert X_transformed.shape[1] == 7

    def test_imputation_strategy_mean(self):
        """Test that mean imputation strategy is used."""
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, _ = build_preprocessor(
            data, 'target', impute_strategy='mean', scale_numeric=False
        )

        # Transform data
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)

        # Mean of [1, 2, 4, 5] = 3.0, so NaN should be replaced with 3.0
        expected_mean = np.mean([1.0, 2.0, 4.0, 5.0])
        assert np.isclose(X_transformed[2, 0], expected_mean)

    def test_imputation_strategy_median(self):
        """Test that median imputation strategy is used."""
        data = pd.DataFrame({
            'feature1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, _ = build_preprocessor(
            data, 'target', impute_strategy='median', scale_numeric=False
        )

        # Transform data
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)

        # Median of [1, 2, 4, 5] = 3.0
        expected_median = np.median([1.0, 2.0, 4.0, 5.0])
        assert np.isclose(X_transformed[2, 0], expected_median)

    def test_categorical_imputation(self):
        """Test that categorical features are imputed with most frequent."""
        data = pd.DataFrame({
            'cat1': ['A', 'B', None, 'A', 'A'],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, _ = build_preprocessor(
            data, 'target', handle_categoricals=True
        )

        # Transform data
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)

        # Should impute with 'A' (most frequent)
        # All rows should be valid (no NaN in output)
        assert not np.isnan(X_transformed).any()

    def test_handle_categoricals_false(self):
        """Test that categorical features are dropped when handle_categoricals=False."""
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'cat': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, _ = build_preprocessor(
            data, 'target', handle_categoricals=False
        )

        # Should only have numeric transformer
        assert 'num' in preprocessor.named_transformers_
        assert 'cat' not in preprocessor.named_transformers_

        # Transform and check shape
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)
        assert X_transformed.shape[1] == 1  # Only numeric feature

    def test_max_categorical_features_limit(self):
        """Test that max_categorical_features parameter works."""
        # Create data with many unique categories
        data = pd.DataFrame({
            'cat': [f'cat_{i}' for i in range(50)],
            'target': [0, 1] * 25
        })

        preprocessor, _ = build_preprocessor(
            data, 'target', handle_categoricals=True, max_categorical_features=10
        )

        # Should limit to 10 categories per feature
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)

        # Should have at most 10 features (limited by max_categories)
        assert X_transformed.shape[1] <= 10

    def test_feature_names_extraction(self):
        """Test that feature names are correctly extracted."""
        data = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50],
            'cat': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, feature_names = build_preprocessor(
            data, 'target', handle_categoricals=True
        )

        # Should extract feature names
        assert feature_names is not None
        assert len(feature_names) > 0

        # Should include numeric column names
        assert 'num1' in feature_names
        assert 'num2' in feature_names

    def test_no_usable_features_error(self):
        """Test error when no features remain after preprocessing."""
        # Only categorical data with handle_categoricals=False
        data = pd.DataFrame({
            'cat1': ['A', 'B', 'A', 'B', 'A'],
            'cat2': ['X', 'Y', 'X', 'Y', 'X'],
            'target': [0, 1, 0, 1, 0]
        })

        with pytest.raises(ValueError, match="No usable features"):
            build_preprocessor(data, 'target', handle_categoricals=False)

    def test_target_not_in_features(self):
        """Test that target column is properly excluded from features."""
        data = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, feature_names = build_preprocessor(
            data, 'target'
        )

        # Transform original data (including target)
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)

        # Should only have 2 features (not including target)
        assert X_transformed.shape[1] == 2

    def test_sparse_output_false(self):
        """Test that OneHotEncoder uses sparse_output=False."""
        data = pd.DataFrame({
            'cat': ['A', 'B', 'C', 'A', 'B'],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, _ = build_preprocessor(
            data, 'target', handle_categoricals=True
        )

        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)

        # Should be a dense array, not sparse
        assert not hasattr(X_transformed, 'toarray')
        assert isinstance(X_transformed, np.ndarray)

    def test_unknown_categories_handling(self):
        """Test that unknown categories are handled gracefully."""
        data = pd.DataFrame({
            'cat': ['A', 'B', 'A', 'B', 'A'],
            'target': [0, 1, 0, 1, 0]
        })

        preprocessor, _ = build_preprocessor(
            data, 'target', handle_categoricals=True
        )

        # Fit on original data
        X = data.drop(columns=['target'])
        preprocessor.fit(X)

        # Transform with new unseen category
        X_new = pd.DataFrame({'cat': ['A', 'C', 'B']})  # 'C' is unseen
        X_transformed = preprocessor.transform(X_new)

        # Should handle unknown category without error
        assert X_transformed.shape[0] == 3

    def test_pandas_3_compatibility(self):
        """Test that the preprocessor uses pandas 3.0 compatible dtypes."""
        data = pd.DataFrame({
            'numeric': [1, 2, 3, 4, 5],
            'string_col': ['a', 'b', 'c', 'd', 'e'],
            'target': [0, 1, 0, 1, 0]
        })

        # Should not raise deprecation warnings about dtypes
        preprocessor, _ = build_preprocessor(
            data, 'target', handle_categoricals=True
        )

        # Should successfully identify and process string columns
        assert 'cat' in preprocessor.named_transformers_

    def test_empty_dataframe_after_target_removal(self):
        """Test handling when only target column exists."""
        data = pd.DataFrame({
            'target': [0, 1, 0, 1, 0]
        })

        with pytest.raises(ValueError, match="No usable features"):
            build_preprocessor(data, 'target')

    def test_all_nan_column_handling(self):
        """Test that columns with all NaN values are handled."""
        data = pd.DataFrame({
            'good_feature': [1, 2, 3, 4, 5],
            'all_nan': [np.nan] * 5,
            'target': [0, 1, 0, 1, 0]
        })

        # Should build preprocessor without error
        preprocessor, _ = build_preprocessor(
            data, 'target'
        )

        # Transform should work
        X = data.drop(columns=['target'])
        X_transformed = preprocessor.transform(X)

        # Should have at least 1 feature (good_feature)
        assert X_transformed.shape[1] >= 1