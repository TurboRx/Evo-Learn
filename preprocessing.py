import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(df: pd.DataFrame,
                       target_column: str,
                       impute_strategy: str = "median",
                       handle_categoricals: bool = True,
                       scale_numeric: bool = True) -> Tuple[Pipeline, list]:
    """
    Build a preprocessing pipeline that imputes, encodes categoricals, and scales numeric features.

    Returns a tuple of (preprocessor_pipeline, feature_names_out)
    """
    X = df.drop(columns=[target_column]) if target_column in df.columns else df.copy()

    # Identify dtypes
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist() if handle_categoricals else []
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    transformers = []

    if numeric_cols:
        num_steps = [("imputer", SimpleImputer(strategy=impute_strategy))]
        if scale_numeric:
            num_steps.append(("scaler", StandardScaler(with_mean=False)))
        transformers.append(("num", Pipeline(steps=num_steps), numeric_cols))

    if handle_categoricals and categorical_cols:
        cat_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])
        transformers.append(("cat", cat_pipeline, categorical_cols))

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
    except Exception:
        feature_names_out = None

    return preprocessor, feature_names_out
