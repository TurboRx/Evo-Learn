"""Data preprocessing utilities with modern Python 3.14 features."""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Configure logging
logger = logging.getLogger(__name__)


def build_preprocessor(
    df: pd.DataFrame,
    target_column: str,
    impute_strategy: str = "median",
    handle_categoricals: bool = True,
    scale_numeric: bool = True,
    max_categorical_features: int = 100
) -> tuple[ColumnTransformer, list[str] | None]:
    """
    Build a preprocessing pipeline that imputes, encodes categoricals, and scales numeric features.

    Args:
        df: Input dataframe
        target_column: Name of target column
        impute_strategy: Imputation strategy for numeric features
        handle_categoricals: Whether to handle categorical features
        scale_numeric: Whether to scale numeric features
        max_categorical_features: Maximum number of one-hot encoded features

    Returns:
        tuple: (preprocessor_pipeline, feature_names_out)
    """
    X = df.drop(columns=[target_column]) if target_column in df.columns else df.copy()

    # Identify dtypes - fix pandas 3.0 deprecation warning
    categorical_cols = (
        X.select_dtypes(include=["object", "category", "string"]).columns.tolist() 
        if handle_categoricals else []
    )
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []

    if numeric_cols:
        num_steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
        if scale_numeric:
            # Use with_mean=True for dense data (default)
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), numeric_cols))

    if handle_categoricals and categorical_cols:
        # Add max_features constraint to prevent memory explosion
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore", 
                sparse_output=False,
                max_categories=max_categorical_features
            ))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))
        logger.info(f"Encoding {len(categorical_cols)} categorical columns (max {max_categorical_features} categories each)")

    if not transformers:
        detected_dtypes = X.dtypes.astype(str).to_dict()
        raise ValueError(
            "No usable features remain after preprocessing. "
            f"handle_categoricals={handle_categoricals}. "
            f"detected_dtypes={detected_dtypes}"
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    # Fit once to get feature names out (optional; actual fitting will be done inside TPOT pipeline)
    preprocessor.fit(X)

    # Try to extract feature names if possible
    feature_names_out = []
    try:
        # Numeric
        feature_names_out += numeric_cols
        # Categorical expanded
        if handle_categoricals and categorical_cols:
            ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cats = ohe.get_feature_names_out(categorical_cols).tolist()
            feature_names_out += cats
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        feature_names_out = None

    return preprocessor, feature_names_out
