"""Professional visualization functions with modern Python 3.14 features."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

# Professional visualization functions


def save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, path: str | Path) -> None:
    """
    Save a Receiver Operating Characteristic (ROC) curve plot with AUC annotation to the given file path.
    
    Parameters:
        y_true (np.ndarray): True binary labels (0 or 1).
        y_proba (np.ndarray): Predicted probabilities or scores for the positive class.
        path (str | Path): Destination file path where the plot will be saved.
    """
    try:
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"ROC curve saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save ROC curve to {path}: {e}")


def save_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, path: str | Path) -> None:
    """
    Save a precision-recall curve plot for binary classification to a file.
    
    Parameters:
        y_true (np.ndarray): True binary labels.
        y_proba (np.ndarray): Predicted probabilities or scores for the positive class.
        path (str | Path): Destination file path for the saved plot.
    """
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(
            recall, precision, color="green", lw=2, label=f"PR curve (AP = {ap:.3f})"
        )
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"PR curve saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save PR curve to {path}: {e}")


def save_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    """
    Create and save a residuals plot comparing predicted values to residuals.
    
    Residuals are computed as y_true - y_pred and plotted against predicted values; the figure is saved to the provided path.
    
    Parameters:
        y_true (np.ndarray | array-like): True target values.
        y_pred (np.ndarray | array-like): Predicted target values.
        path (str | Path): File path where the plot image will be saved.
    """
    try:
        residuals = np.array(y_true) - np.array(y_pred)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_pred, residuals, s=12, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Predicted")
        plt.ylabel("Residuals (y - y_pred)")
        plt.title("Residuals Plot")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"Residuals plot saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save residuals plot to {path}: {e}")


def save_actual_vs_pred(
    y_true: np.ndarray, y_pred: np.ndarray, path: str | Path
) -> None:
    """
    Create and save a scatter plot comparing actual values to predicted values with a 1:1 reference line.
    
    Parameters:
        y_true (np.ndarray): Array of true target values.
        y_pred (np.ndarray): Array of predicted values corresponding to `y_true`.
        path (str | Path): File path where the plot image will be saved.
    """
    try:
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, s=12, alpha=0.7)
        vmin = min(np.min(y_true), np.min(y_pred))
        vmax = max(np.max(y_true), np.max(y_pred))
        plt.plot([vmin, vmax], [vmin, vmax], color="red", linestyle="--", lw=1)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"Actual vs predicted plot saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save actual vs predicted plot to {path}: {e}")


def plot_feature_distributions(
    data: pd.DataFrame, target_column: str, output_dir: str | Path
) -> None:
    """
    Generate and save distribution plots for up to 10 numeric features, segmented by the specified target column.
    
    Each plot is saved as a PNG file named `dist_<feature>.png` in `output_dir`. If more than 10 numeric features exist, only the first 10 are plotted and a warning is logged.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing features and the target column.
        target_column (str): Name of the column used to split/colour distributions (hue).
        output_dir (str | Path): Directory path where PNG files will be written.
    """
    try:
        import seaborn as sns

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        # Create distribution plots
        num_plots = min(len(numeric_cols), 10)  # Limit to first 10 features
        for col in numeric_cols[:num_plots]:
            plt.figure(figsize=(8, 6))
            sns.histplot(data=data, x=col, hue=target_column, kde=True)
            plt.title(f"Distribution of {col}")
            plt.tight_layout()
            plt.savefig(out_path / f"dist_{col}.png")
            plt.close()

        if len(numeric_cols) > 10:
            logger.warning(
                f"Only plotted first 10 of {len(numeric_cols)} numeric features"
            )
        logger.info(f"Feature distribution plots saved to {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create feature distribution plots: {e}")


def plot_correlation_matrix(data: pd.DataFrame, output_path: str | Path) -> None:
    """
    Generate and save a heatmap of the correlation matrix for the DataFrame's numeric columns.
    
    Parameters:
        data (pd.DataFrame): Input data containing features to correlate.
        output_path (str | Path): File path where the heatmap image will be saved.
    
    Notes:
        If the DataFrame contains fewer than two numeric columns, no heatmap is created and a warning is logged.
    """
    try:
        import seaborn as sns

        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])

        if numeric_data.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_data.corr()
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap="coolwarm",
                center=0,
                square=True,
                fmt=".2f",
                cbar_kws={"shrink": 0.8},
            )
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Correlation matrix saved to {output_path}")
        else:
            logger.warning("Not enough numeric columns for correlation matrix")
    except Exception as e:
        logger.error(f"Failed to create correlation matrix: {e}")


def create_evaluation_dashboard(
    results: dict[str, Any], output_dir: str | Path
) -> None:
    """
    Create a simple HTML evaluation dashboard summarizing numeric metrics and save it to disk.
    
    The function ensures the output directory exists, formats numeric entries found under the "metrics" key
    to four decimal places, and writes an "evaluation_report.html" file into the provided directory.
    
    Parameters:
        results (dict[str, Any]): Evaluation data; if a "metrics" key is present, numeric metric values will be rendered.
        output_dir (str | Path): Directory where "evaluation_report.html" will be written.
    """
    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # Create summary HTML report
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Evo-Learn Model Evaluation</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .metric {{ background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .header {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            </style>
        </head>
        <body>
            <h1 class="header">Model Evaluation Results</h1>
            <div class="metrics">
        """

        if "metrics" in results:
            for metric, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    html_content += f'<div class="metric"><strong>{metric.title()}:</strong> {value:.4f}</div>'

        html_content += """
            </div>
        </body>
        </html>
        """

        (out_path / "evaluation_report.html").write_text(html_content)
        logger.info("Evaluation dashboard saved to %s", output_dir)

    except Exception as e:
        logger.error(f"Failed to create evaluation dashboard: {e}")