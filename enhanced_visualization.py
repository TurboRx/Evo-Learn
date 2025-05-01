import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from typing import Dict, List, Tuple, Optional, Any, Union
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def plot_feature_distributions(data: pd.DataFrame, target_column: str, 
                              numerical_cols: List[str] = None, 
                              categorical_cols: List[str] = None,
                              output_dir: str = 'visualizations'):
    """
    Create distribution plots for features in the dataset.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
        target_column (str): Name of the target variable column.
        numerical_cols (List[str], optional): List of numerical columns to plot.
        categorical_cols (List[str], optional): List of categorical columns to plot.
        output_dir (str): Directory to save visualization outputs.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unique target values for colors
    target_values = data[target_column].unique()
    
    # If column lists are not provided, infer them
    if numerical_cols is None:
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        # Remove target if it's numerical
        if target_column in numerical_cols:
            numerical_cols.remove(target_column)
    
    if categorical_cols is None:
        categorical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        # Remove target if it's categorical
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
    
    # Set visualization style
    sns.set(style="whitegrid")
    
    # Plot numerical features
    if numerical_cols:
        logger.info(f"Creating distribution plots for {len(numerical_cols)} numerical features")
        for col in numerical_cols:
            try:
                plt.figure(figsize=(10, 6))
                
                # Create distribution plot for each target class
                for value in target_values:
                    subset = data[data[target_column] == value]
                    sns.kdeplot(subset[col], label=f'{target_column}={value}')
                
                plt.title(f'Distribution of {col} by {target_column}')
                plt.xlabel(col)
                plt.ylabel('Density')
                plt.legend()
                
                # Save the figure
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'dist_{col}.png'))
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating distribution plot for {col}: {e}")
    
    # Plot categorical features
    if categorical_cols:
        logger.info(f"Creating bar plots for {len(categorical_cols)} categorical features")
        for col in categorical_cols:
            try:
                plt.figure(figsize=(12, 6))
                
                # Create countplot
                ax = sns.countplot(x=col, hue=target_column, data=data)
                
                # Improve readability for many categories
                if data[col].nunique() > 10:
                    plt.xticks(rotation=45, ha='right')
                
                plt.title(f'Distribution of {col} by {target_column}')
                plt.tight_layout()
                
                # Save the figure
                plt.savefig(os.path.join(output_dir, f'bar_{col}.png'))
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating bar plot for {col}: {e}")
    
    logger.info(f"All distribution plots saved to {output_dir}")

def plot_correlation_matrix(data: pd.DataFrame, numerical_cols: List[str] = None, 
                           output_path: str = 'visualizations/correlation_matrix.png'):
    """
    Create and save a correlation matrix heatmap.
    
    Args:
        data (pd.DataFrame): Input DataFrame.
        numerical_cols (List[str], optional): List of numerical columns to include.
        output_path (str): Path to save the visualization.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If column list is not provided, use all numerical columns
    if numerical_cols is None:
        numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Check if we have enough numerical columns
    if len(numerical_cols) < 2:
        logger.warning("Not enough numerical columns for correlation analysis")
        return
    
    try:
        # Compute correlation matrix
        correlation = data[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation, dtype=bool))  # Create a mask for the upper triangle
        
        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(correlation, mask=mask, cmap=cmap, annot=True, fmt=".2f", 
                   center=0, square=True, linewidths=.5)
        
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Correlation matrix saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating correlation matrix: {e}")

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str] = None,
                         output_path: str = 'visualizations/confusion_matrix.png'):
    """
    Create and save a confusion matrix visualization.
    
    Args:
        y_true (np.ndarray): True target values.
        y_pred (np.ndarray): Predicted target values.
        class_names (List[str], optional): Names for the classes.
        output_path (str): Path to save the visualization.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # If class names aren't provided, use default names
        if class_names is None:
            class_names = [f'Class {i}' for i in range(len(np.unique(y_true)))]
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Confusion matrix saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating confusion matrix plot: {e}")

def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, 
                  output_path: str = 'visualizations/roc_curve.png'):
    """
    Create and save a ROC curve visualization for binary classification.
    
    Args:
        y_true (np.ndarray): True binary labels.
        y_score (np.ndarray): Target scores (probability estimates of the positive class).
        output_path (str): Path to save the visualization.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Check if binary classification
        if len(np.unique(y_true)) != 2:
            logger.warning("ROC curve requires binary classification problem")
            return
        
        # Compute ROC curve and ROC area
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        
        # Create the plot
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"ROC curve saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating ROC curve plot: {e}")

def plot_learning_curve(estimator: Any, X: np.ndarray, y: np.ndarray, 
                       output_path: str = 'visualizations/learning_curve.png',
                       cv: int = 5, train_sizes: np.ndarray = None):
    """
    Create and save a learning curve visualization.
    
    Args:
        estimator: A classifier or regressor instance.
        X (np.ndarray): Training vector.
        y (np.ndarray): Target values.
        output_path (str): Path to save the visualization.
        cv (int): Number of cross-validation folds.
        train_sizes (np.ndarray, optional): Array of training sizes.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        from sklearn.model_selection import learning_curve
        
        # Default train sizes if not provided
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 5)
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=-1, train_sizes=train_sizes
        )
        
        # Calculate means and standard deviations
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot the learning curve
        plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Score')
        plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
        
        plt.plot(train_sizes, test_mean, color='green', marker='s', markersize=5, label='Validation Score')
        plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
        
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Learning curve saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating learning curve plot: {e}")

def plot_feature_importance(feature_names: List[str], feature_importances: np.ndarray, 
                           output_path: str = 'visualizations/feature_importance.png',
                           top_n: int = None):
    """
    Create and save a feature importance visualization.
    
    Args:
        feature_names (List[str]): Names of the features.
        feature_importances (np.ndarray): Importance scores for each feature.
        output_path (str): Path to save the visualization.
        top_n (int, optional): Only show top N features by importance.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        # Validation
        if len(feature_names) != len(feature_importances):
            raise ValueError("Length of feature_names and feature_importances must match")
        
        # Sort features by importance
        indices = np.argsort(feature_importances)
        
        # Limit to top N if specified
        if top_n is not None and top_n < len(indices):
            indices = indices[-top_n:]
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Create horizontal bar chart
        plt.barh(range(len(indices)), feature_importances[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Feature importance plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating feature importance plot: {e}")

def create_evaluation_dashboard(results: Dict[str, Any], output_dir: str = 'visualizations'):
    """
    Create a comprehensive evaluation dashboard with multiple visualizations.
    
    Args:
        results (Dict[str, Any]): Dictionary containing model evaluation results.
        output_dir (str): Directory to save visualization outputs.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Extract data from results
        y_true = results.get('y_true')
        y_pred = results.get('y_pred')
        y_proba = results.get('y_proba')
        feature_names = results.get('feature_names')
        feature_importance = results.get('feature_importances')
        metrics = results.get('metrics')
        model_info = results.get('model_info', {})
        
        # Check if we have necessary data
        if y_true is None or y_pred is None:
            logger.warning("Missing required y_true or y_pred for evaluation dashboard")
            return
        
        # 1. Create confusion matrix
        if 'classification' in model_info.get('task', ''):
            plot_confusion_matrix(
                y_true, y_pred,
                class_names=results.get('class_names'),
                output_path=os.path.join(output_dir, 'confusion_matrix.png')
            )
        
        # 2. Create ROC curve for binary classification
        if y_proba is not None and len(np.unique(y_true)) == 2:
            plot_roc_curve(
                y_true, y_proba,
                output_path=os.path.join(output_dir, 'roc_curve.png')
            )
        
        # 3. Create feature importance plot if available
        if feature_names is not None and feature_importance is not None:
            plot_feature_importance(
                feature_names, feature_importance,
                output_path=os.path.join(output_dir, 'feature_importance.png')
            )
        
        # 4. Create a summary metrics text file
        if metrics:
            metrics_path = os.path.join(output_dir, 'metrics_summary.txt')
            with open(metrics_path, 'w') as f:
                f.write("MODEL EVALUATION METRICS\n")
                f.write("========================\n\n")
                
                # Write model info
                f.write("MODEL INFORMATION:\n")
                for key, value in model_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
                
                # Write metrics
                f.write("PERFORMANCE METRICS:\n")
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
            
            logger.info(f"Metrics summary saved to {metrics_path}")
        
        logger.info(f"Evaluation dashboard created in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error creating evaluation dashboard: {e}")