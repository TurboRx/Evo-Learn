import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any

# Existing visualization functions are assumed present. Adding new helpers:

def save_roc_curve(y_true, y_proba, path: str):
    try:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception:
        pass

def save_pr_curve(y_true, y_proba, path: str):
    try:
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(recall, precision, color='green', lw=2, label=f'PR curve (AP = {ap:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception:
        pass

def save_residuals(y_true, y_pred, path: str):
    try:
        residuals = np.array(y_true) - np.array(y_pred)
        plt.figure(figsize=(6, 5))
        plt.scatter(y_pred, residuals, s=12, alpha=0.7)
        plt.axhline(0, color='red', linestyle='--', lw=1)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals (y - y_pred)')
        plt.title('Residuals Plot')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception:
        pass

def save_actual_vs_pred(y_true, y_pred, path: str):
    try:
        plt.figure(figsize=(6, 5))
        plt.scatter(y_true, y_pred, s=12, alpha=0.7)
        vmin = min(np.min(y_true), np.min(y_pred))
        vmax = max(np.max(y_true), np.max(y_pred))
        plt.plot([vmin, vmax], [vmin, vmax], color='red', linestyle='--', lw=1)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception:
        pass

# Minimal dashboard wrapper to keep compatibility

def create_evaluation_dashboard(results: Dict[str, Any], output_dir: str):
    # Stub: assumes existing dashboard logic in file; keep as thin wrapper
    os.makedirs(output_dir, exist_ok=True)
    # You can extend to write a quick HTML/Markdown summary here if desired.
    pass
