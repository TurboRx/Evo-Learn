import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Dict, Any

# Professional visualization functions

def save_roc_curve(y_true, y_proba, path: str):
    """
    Save ROC curve plot to file.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        path: Output file path
    """
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
    """
    Save Precision-Recall curve plot to file.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        path: Output file path
    """
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
    """
    Save residuals plot for regression models.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        path: Output file path
    """
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
    """
    Save actual vs predicted values plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        path: Output file path
    """
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

def plot_feature_distributions(data, target_column: str, output_dir: str):
    """
    Create feature distribution plots.
    
    Args:
        data: Input dataframe
        target_column: Name of target column
        output_dir: Directory to save plots
    """
    try:
        import seaborn as sns
        os.makedirs(output_dir, exist_ok=True)
        
        # Get numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        
        # Create distribution plots
        for col in numeric_cols[:10]:  # Limit to first 10 features
            plt.figure(figsize=(8, 6))
            sns.histplot(data=data, x=col, hue=target_column, kde=True)
            plt.title(f'Distribution of {col}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'dist_{col}.png'))
            plt.close()
    except Exception:
        pass

def plot_correlation_matrix(data, output_path: str):
    """
    Create correlation matrix heatmap.
    
    Args:
        data: Input dataframe
        output_path: Output file path
    """
    try:
        import seaborn as sns
        
        # Get numeric columns only
        numeric_data = data.select_dtypes(include=[np.number])
        
        if numeric_data.shape[1] > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = numeric_data.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
    except Exception:
        pass

def create_evaluation_dashboard(results: Dict[str, Any], output_dir: str):
    """
    Create evaluation dashboard with summary plots.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save dashboard files
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create summary HTML report
        html_content = f"""
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
        
        if 'metrics' in results:
            for metric, value in results['metrics'].items():
                if isinstance(value, (int, float)):
                    html_content += f'<div class="metric"><strong>{metric.title()}:</strong> {value:.4f}</div>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(output_dir, 'evaluation_report.html'), 'w') as f:
            f.write(html_content)
            
    except Exception:
        pass