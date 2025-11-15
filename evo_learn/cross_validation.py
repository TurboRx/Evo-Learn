"""Cross-validation utilities for robust model evaluation."""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    cv_folds: int = 5,
    random_state: int = 42,
    scoring: Optional[str] = None
) -> Dict[str, Any]:
    """Perform k-fold cross-validation.
    
    Args:
        model: Model pipeline to evaluate
        X: Feature data
        y: Target data
        task: 'classification' or 'regression'
        cv_folds: Number of folds
        random_state: Random seed
        scoring: Primary scoring metric (optional)
        
    Returns:
        Dict containing cross-validation results
    """
    logger.info(f"Starting {cv_folds}-fold cross-validation...")
    
    # Choose appropriate splitter
    if task.lower() == "classification":
        splitter = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state
        )
    else:
        splitter = KFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state
        )
    
    # Storage for fold results
    fold_scores = []
    fold_predictions = []
    
    # Perform cross-validation
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X, y)):
        logger.info(f"Processing fold {fold_idx + 1}/{cv_folds}...")
        
        # Split data
        X_train_fold = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val_fold = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]
        
        # Clone and train model
        fold_model = clone(model)
        fold_model.fit(X_train_fold, y_train_fold)
        
        # Make predictions
        y_pred = fold_model.predict(X_val_fold)
        
        # Compute metrics
        if task.lower() == "classification":
            # Get probabilities for ROC AUC
            y_proba = None
            if hasattr(fold_model, 'predict_proba') and len(np.unique(y)) == 2:
                try:
                    y_proba = fold_model.predict_proba(X_val_fold)[:, 1]
                except Exception:
                    pass
            
            scores = {
                'accuracy': accuracy_score(y_val_fold, y_pred),
                'precision': precision_score(y_val_fold, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_val_fold, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_val_fold, y_pred, average='weighted', zero_division=0)
            }
            
            if y_proba is not None:
                try:
                    scores['roc_auc'] = roc_auc_score(y_val_fold, y_proba)
                except Exception:
                    pass
        else:
            mse = mean_squared_error(y_val_fold, y_pred)
            scores = {
                'mse': mse,
                'rmse': np.sqrt(mse),
                'mae': mean_absolute_error(y_val_fold, y_pred),
                'r2': r2_score(y_val_fold, y_pred)
            }
        
        fold_scores.append(scores)
        fold_predictions.append({
            'y_true': y_val_fold,
            'y_pred': y_pred,
            'indices': val_idx
        })
        
        logger.info(f"Fold {fold_idx + 1} scores: {scores}")
    
    # Aggregate results
    metric_names = fold_scores[0].keys()
    aggregated_scores = {}
    
    for metric in metric_names:
        values = [fold[metric] for fold in fold_scores]
        aggregated_scores[f"{metric}_mean"] = np.mean(values)
        aggregated_scores[f"{metric}_std"] = np.std(values)
        aggregated_scores[f"{metric}_min"] = np.min(values)
        aggregated_scores[f"{metric}_max"] = np.max(values)
    
    results = {
        'cv_scores': aggregated_scores,
        'fold_scores': fold_scores,
        'fold_predictions': fold_predictions,
        'n_folds': cv_folds
    }
    
    logger.info("Cross-validation completed")
    logger.info(f"Aggregated scores: {aggregated_scores}")
    
    return results


def nested_cross_validation(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task: str = "classification",
    outer_folds: int = 5,
    inner_folds: int = 3,
    random_state: int = 42
) -> Dict[str, Any]:
    """Perform nested cross-validation for unbiased performance estimation.
    
    Args:
        model: Model pipeline to evaluate
        X: Feature data
        y: Target data
        task: 'classification' or 'regression'
        outer_folds: Number of outer folds
        inner_folds: Number of inner folds for hyperparameter tuning
        random_state: Random seed
        
    Returns:
        Dict containing nested CV results
    """
    logger.info(
        f"Starting nested cross-validation: "
        f"outer={outer_folds}, inner={inner_folds}"
    )
    
    # Outer loop for performance estimation
    if task.lower() == "classification":
        outer_splitter = StratifiedKFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=random_state
        )
    else:
        outer_splitter = KFold(
            n_splits=outer_folds,
            shuffle=True,
            random_state=random_state
        )
    
    outer_scores = []
    
    for outer_idx, (train_idx, test_idx) in enumerate(outer_splitter.split(X, y)):
        logger.info(f"Outer fold {outer_idx + 1}/{outer_folds}")
        
        # Split data
        X_train_outer = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_test_outer = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
        y_train_outer = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_test_outer = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
        
        # Inner cross-validation for model selection/tuning
        inner_results = cross_validate_model(
            model=model,
            X=X_train_outer,
            y=y_train_outer,
            task=task,
            cv_folds=inner_folds,
            random_state=random_state
        )
        
        # Train final model on all outer training data
        final_model = clone(model)
        final_model.fit(X_train_outer, y_train_outer)
        
        # Evaluate on outer test set
        y_pred = final_model.predict(X_test_outer)
        
        if task.lower() == "classification":
            scores = {
                'accuracy': accuracy_score(y_test_outer, y_pred),
                'f1': f1_score(y_test_outer, y_pred, average='weighted', zero_division=0)
            }
        else:
            mse = mean_squared_error(y_test_outer, y_pred)
            scores = {
                'rmse': np.sqrt(mse),
                'r2': r2_score(y_test_outer, y_pred)
            }
        
        outer_scores.append(scores)
        logger.info(f"Outer fold {outer_idx + 1} scores: {scores}")
    
    # Aggregate outer scores
    metric_names = outer_scores[0].keys()
    final_scores = {}
    
    for metric in metric_names:
        values = [fold[metric] for fold in outer_scores]
        final_scores[f"{metric}_mean"] = np.mean(values)
        final_scores[f"{metric}_std"] = np.std(values)
    
    results = {
        'nested_cv_scores': final_scores,
        'outer_fold_scores': outer_scores,
        'n_outer_folds': outer_folds,
        'n_inner_folds': inner_folds
    }
    
    logger.info("Nested cross-validation completed")
    logger.info(f"Final scores: {final_scores}")
    
    return results
