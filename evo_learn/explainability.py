"""Model explainability and interpretation with SHAP."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Optional, Any, Tuple
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explain model predictions using SHAP values."""
    
    def __init__(self, model: Any, X_train: pd.DataFrame):
        """Initialize explainer.
        
        Args:
            model: Trained model pipeline
            X_train: Training data for background samples
        """
        if not SHAP_AVAILABLE:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            )
        
        self.model = model
        self.X_train = X_train
        self.explainer = None
        self.shap_values = None
        self._X_test_transformed = None
        self._transformed_feature_names: list | None = None
        
        logger.info("Initialized ModelExplainer")
    
    def _get_predict_function(self):
        """Get appropriate prediction function for SHAP."""
        # Handle sklearn pipelines
        if hasattr(self.model, 'predict_proba'):
            return lambda x: self.model.predict_proba(x)
        else:
            return lambda x: self.model.predict(x)
    
    def compute_shap_values(
        self,
        X_test: pd.DataFrame,
        max_samples: int = 100,
        explainer_type: str = "auto"
    ) -> np.ndarray:
        """Compute SHAP values for test data.
        
        Args:
            X_test: Test data to explain
            max_samples: Maximum samples to use (for performance)
            explainer_type: Type of explainer ('tree', 'kernel', 'linear', 'auto')
            
        Returns:
            SHAP values array
        """
        # Limit samples for performance
        if len(X_test) > max_samples:
            logger.info(f"Limiting explanation to {max_samples} samples")
            X_test = X_test.sample(n=max_samples, random_state=42)
        
        # Transform data through preprocessing
        if hasattr(self.model, 'named_steps'):
            steps = list(self.model.named_steps.items())
            estimator_step = next(
                (step for step in reversed(steps) if hasattr(step[1], "predict") or hasattr(step[1], "predict_proba")),
                None,
            )
            if estimator_step is None:
                raise ValueError("Pipeline is missing an estimator step with predict/predict_proba.")
            estimator_name, model_to_explain = estimator_step

            preprocess_step = next(
                (
                    step
                    for step in steps
                    if step[0] != estimator_name and hasattr(step[1], "transform")
                ),
                None,
            )
            if preprocess_step is None:
                X_test_transformed = X_test
                X_train_transformed = self.X_train.sample(
                    min(100, len(self.X_train)), random_state=42
                )
            else:
                X_test_transformed = preprocess_step[1].transform(X_test)
                X_train_transformed = preprocess_step[1].transform(
                    self.X_train.sample(min(100, len(self.X_train)), random_state=42)
                )
                try:
                    self._transformed_feature_names = list(
                        preprocess_step[1].get_feature_names_out()
                    )
                except Exception:
                    pass
        else:
            X_test_transformed = X_test
            X_train_transformed = self.X_train.sample(min(100, len(self.X_train)), random_state=42)
            model_to_explain = self.model
        
        self._X_test_transformed = X_test_transformed
        # Use transformed feature names if already set (by preprocessor), else fall back
        # to column names of the transformed data (works for both ndarray and DataFrame)
        if self._transformed_feature_names is None:
            self._transformed_feature_names = (
                list(X_test_transformed.columns)
                if hasattr(X_test_transformed, "columns")
                else None
            )
        
        # Choose appropriate explainer
        try:
            if explainer_type == "auto":
                # Try tree explainer first (fastest)
                try:
                    self.explainer = shap.TreeExplainer(model_to_explain)
                    explainer_type = "tree"
                except Exception:
                    # Fall back to kernel explainer
                    self.explainer = shap.KernelExplainer(
                        model_to_explain.predict,
                        X_train_transformed
                    )
                    explainer_type = "kernel"
            elif explainer_type == "tree":
                self.explainer = shap.TreeExplainer(model_to_explain)
            elif explainer_type == "kernel":
                self.explainer = shap.KernelExplainer(
                    model_to_explain.predict,
                    X_train_transformed
                )
            elif explainer_type == "linear":
                self.explainer = shap.LinearExplainer(
                    model_to_explain,
                    X_train_transformed
                )
            else:
                raise ValueError(f"Unknown explainer type: {explainer_type}")
            
            logger.info(f"Using {explainer_type} explainer")
            
            # Compute SHAP values
            self.shap_values = self.explainer.shap_values(X_test_transformed)
            
            logger.info("SHAP values computed successfully")
            return self.shap_values
            
        except Exception as e:
            logger.error(f"Failed to compute SHAP values: {e}")
            raise
    
    def plot_summary(
        self,
        output_path: Optional[str] = None,
        plot_type: str = "dot",
        max_display: int = 20
    ) -> None:
        """Create SHAP summary plot.
        
        Args:
            output_path: Path to save plot
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            self.shap_values,
            features=self._X_test_transformed,
            feature_names=self._transformed_feature_names,
            plot_type=plot_type,
            max_display=max_display,
            show=False
        )
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"SHAP summary plot saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(
        self,
        output_path: Optional[str] = None,
        max_display: int = 20
    ) -> None:
        """Create mean absolute SHAP value plot (feature importance).
        
        Args:
            output_path: Path to save plot
            max_display: Maximum features to display
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        # Handle multi-output SHAP values
        if isinstance(self.shap_values, list):
            shap_vals = np.abs(self.shap_values[1])  # Use positive class for binary
        else:
            shap_vals = np.abs(self.shap_values)
        
        # Compute mean importance
        importance = np.mean(shap_vals, axis=0)
        
        # Get feature names (use transformed names to match SHAP values dimensions)
        feature_names = (
            self._transformed_feature_names
            if self._transformed_feature_names is not None
            else [f"Feature {i}" for i in range(len(importance))]
        )
        
        # Sort and plot
        indices = np.argsort(importance)[-max_display:]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Mean |SHAP value| (Average Impact on Model Output)')
        plt.title('Feature Importance (SHAP)')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Feature importance plot saved: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def explain_prediction(
        self,
        sample_idx: int,
        output_path: Optional[str] = None
    ) -> None:
        """Create waterfall plot for a single prediction.
        
        Args:
            sample_idx: Index of sample to explain
            output_path: Path to save plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed. Call compute_shap_values first.")
        
        # Handle multi-output SHAP values
        if isinstance(self.shap_values, list):
            class_idx = 1  # positive class for binary; last class for multiclass
            class_idx = min(class_idx, len(self.shap_values) - 1)
            shap_vals = self.shap_values[class_idx][sample_idx]
        else:
            class_idx = None
            shap_vals = self.shap_values[sample_idx]
        
        # Use iloc for positional indexing so non-sequential DataFrame indices
        # (e.g. after .sample()) don't cause a KeyError.
        if self._X_test_transformed is not None:
            transformed = self._X_test_transformed
            if hasattr(transformed, "iloc"):
                sample_data = transformed.iloc[sample_idx]
            else:
                sample_data = transformed[sample_idx]
        else:
            sample_data = None

        # Resolve base_values to a scalar when expected_value is array-like
        # (multi-class explainers return one expected value per class).
        raw_ev = self.explainer.expected_value if hasattr(self.explainer, 'expected_value') else 0
        if hasattr(raw_ev, '__len__'):
            if class_idx is not None and class_idx < len(raw_ev):
                base_value = float(raw_ev[class_idx])
            else:
                base_value = float(raw_ev[0])
        else:
            base_value = float(raw_ev) if raw_ev != 0 else 0

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals,
                base_values=base_value,
                data=sample_data,
                feature_names=self._transformed_feature_names
            ),
            show=False
        )
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Prediction explanation saved: {output_path}")
        else:
            plt.show()
        
        plt.close()


def explain_model(
    model: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    output_dir: str,
    model_name: str = "model"
) -> None:
    """Generate comprehensive model explanations.
    
    Args:
        model: Trained model pipeline
        X_train: Training data
        X_test: Test data
        output_dir: Directory to save plots
        model_name: Name prefix for saved files
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping explainability analysis")
        return
    
    try:
        logger.info("Generating model explanations with SHAP...")
        
        # Create explainer
        explainer = ModelExplainer(model, X_train)
        
        # Compute SHAP values
        explainer.compute_shap_values(X_test, max_samples=100)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate plots
        explainer.plot_summary(
            output_path=str(output_path / f"{model_name}_shap_summary.png")
        )
        
        explainer.plot_feature_importance(
            output_path=str(output_path / f"{model_name}_feature_importance.png")
        )
        
        # Explain first few predictions — use the number of rows in the stored
        # SHAP values (which may have been sampled) rather than len(X_test), so
        # the indices are always valid positional offsets into _X_test_transformed.
        n_explained = len(
            explainer.shap_values[0]
            if isinstance(explainer.shap_values, list)
            else explainer.shap_values
        )
        for i in range(min(3, n_explained)):
            explainer.explain_prediction(
                sample_idx=i,
                output_path=str(output_path / f"{model_name}_explanation_{i}.png")
            )
        
        logger.info(f"Model explanations saved to: {output_dir}")
        
    except Exception as e:
        logger.warning(f"Could not generate model explanations: {e}")
