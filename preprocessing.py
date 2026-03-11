"""Data preprocessing utilities."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def build_preprocessor(
    df: pd.DataFrame,
    target_column: str,
    impute_strategy: str = "median",
    handle_categoricals: bool = True,
    scale_numeric: bool = True,
    max_categorical_features: int = 100,
) -> tuple[ColumnTransformer, list[str] | None]:
    """Build preprocessing pipeline with imputation, encoding, and scaling."""
    X = df.drop(columns=[target_column]) if target_column in df.columns else df.copy()

    categorical_cols = (
        X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if handle_categoricals
        else []
    )
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []

    if numeric_cols:
        num_steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(steps=num_steps), numeric_cols))

    if handle_categoricals and categorical_cols:
        # Filter out columns that are all NaN or have no valid values
        valid_cat_cols = [
            col for col in categorical_cols
            if not X[col].isna().all() and X[col].nunique(dropna=True) > 0
        ]

        if valid_cat_cols:
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                            max_categories=max_categorical_features,
                        ),
                    ),
                ]
            )
            transformers.append(("cat", cat_pipeline, valid_cat_cols))
            logger.info(
                f"Encoding {len(valid_cat_cols)} categorical columns "
                f"(max {max_categorical_features} categories each)"
            )
        elif categorical_cols:
            logger.warning(
                f"Found {len(categorical_cols)} categorical columns, but all are empty or have no valid values. Skipping."
            )

    if not transformers:
        raise ValueError(
            f"No usable features. handle_categoricals={handle_categoricals}, "
            f"dtypes={X.dtypes.astype(str).to_dict()}"
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    preprocessor.fit(X)

    feature_names_out: list[str] | None = None
    try:
        feature_names_out = numeric_cols.copy()
        if handle_categoricals and "cat" in preprocessor.named_transformers_:
            ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
            # Use the valid_cat_cols that were actually processed
            cat_cols_used = [t[2] for t in transformers if t[0] == "cat"]
            if cat_cols_used:
                cats = ohe.get_feature_names_out(cat_cols_used[0]).tolist()
                feature_names_out += cats
    except Exception as e:
        logger.warning(f"Feature name extraction failed: {e}")
        feature_names_out = None

    return preprocessor, feature_names_out
