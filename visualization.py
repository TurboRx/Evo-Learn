"""Visualization functions."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def safe_filename(filename: str, max_length: int = 200) -> str:
    """Sanitize filename for filesystem safety."""
    sanitized = re.sub(r'[/\\:*?"<>|]', "_", str(filename))
    sanitized = re.sub(r"[\s_]+", "_", sanitized)
    sanitized = sanitized.strip(". ")
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
    return sanitized if sanitized else "unnamed"


def save_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, path: str | Path) -> None:
    """Save ROC curve plot."""
    try:
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"ROC saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save ROC curve: {e}")


def save_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, path: str | Path) -> None:
    """Save Precision-Recall curve plot."""
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score

        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color="green", lw=2, label=f"PR (AP={ap:.3f})")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"PR saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save PR curve: {e}")


def save_residuals(y_true: np.ndarray, y_pred: np.ndarray, path: str | Path) -> None:
    """Save residuals plot for regression."""
    try:
        residuals = np.array(y_true) - np.array(y_pred)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_pred, residuals, s=12, alpha=0.7)
        plt.axhline(0, color="red", linestyle="--", lw=1)
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residuals Plot")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        logger.info(f"Residuals saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save residuals plot: {e}")


def save_actual_vs_pred(
    y_true: np.ndarray, y_pred: np.ndarray, path: str | Path
) -> None:
    """Save actual vs predicted plot."""
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
        logger.info(f"Actual vs predicted saved: {path}")
    except Exception as e:
        logger.error(f"Failed to save actual vs predicted plot: {e}")


def plot_feature_distributions(
    data: pd.DataFrame, target_column: str, output_dir: str | Path
) -> None:
    """Create feature distribution plots."""
    try:
        import seaborn as sns

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)

        num_plots = min(len(numeric_cols), 10)
        for col in numeric_cols[:num_plots]:
            safe_col_name = safe_filename(col)
            output_file = out_path / f"dist_{safe_col_name}.png"

            try:
                output_file_resolved = output_file.resolve()
                out_path_resolved = out_path.resolve()
                if not str(output_file_resolved).startswith(str(out_path_resolved)):
                    logger.error(f"Path traversal detected: {col}")
                    continue
            except (OSError, RuntimeError) as e:
                logger.error(f"Path resolution error for {col}: {e}")
                continue

            plt.figure(figsize=(8, 6))
            sns.histplot(data=data, x=col, hue=target_column, kde=True)
            plt.title(f"Distribution: {col}")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()

        if len(numeric_cols) > 10:
            logger.warning(f"Plotted 10/{len(numeric_cols)} features")
        logger.info(f"Distributions saved: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create distribution plots: {e}")


def plot_correlation_matrix(data: pd.DataFrame, output_path: str | Path) -> None:
    """Create correlation matrix heatmap."""
    try:
        import seaborn as sns

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
            plt.title("Feature Correlation")
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            logger.info(f"Correlation matrix saved: {output_path}")
        else:
            logger.warning("Not enough numeric columns for correlation matrix (need >1)")
    except Exception as e:
        logger.error(f"Failed to create correlation matrix: {e}")


def create_evaluation_dashboard(
    results: dict[str, Any], output_dir: str | Path
) -> None:
    """Create evaluation dashboard."""
    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation</title>
            <style>
                body { font-family: Arial; margin: 40px; }
                .metric { background: #f5f5f5; padding: 10px; margin: 5px 0; border-radius: 5px; }
                .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
            </style>
        </head>
        <body>
            <h1 class="header">Model Evaluation</h1>
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
        logger.info("Dashboard saved: %s", output_dir)

    except Exception as e:
        logger.error(f"Failed to create dashboard: {e}")
