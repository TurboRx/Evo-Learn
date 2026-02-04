#!/usr/bin/env python3
"""
Command Line Interface for Evo-Learn AutoML Toolkit

Professional CLI with comprehensive functionality for training, prediction,
evaluation, and visualization of machine learning models.
Uses modern Python 3.14 features including pattern matching.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evo_learn.log')
    ]
)
logger = logging.getLogger(__name__)

def setup_argparse():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Evo-Learn: Professional AutoML Toolkit',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model using AutoML')
    train_parser.add_argument('--data', required=True, help='Path to CSV data file')
    train_parser.add_argument('--target', required=True, help='Name of target column')
    train_parser.add_argument('--task', choices=['classification', 'regression'], default=None,
                             help='Machine learning task (defaults to config default_task or classification)')
    train_parser.add_argument('--generations', type=int, default=5, help='TPOT generations')
    train_parser.add_argument('--population', type=int, default=20, help='TPOT population size')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test split size')
    train_parser.add_argument('--output-dir', default='models', help='Model output directory')
    train_parser.add_argument('--max-time', type=int, default=None, help='Total TPOT time (mins)')
    train_parser.add_argument('--max-eval-time', type=int, default=5, help='Per-pipeline eval time (mins)')
    train_parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    train_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    # Configuration & preprocessing
    train_parser.add_argument('--config', help='Path to config.yaml')
    train_parser.add_argument('--no-categoricals', action='store_true', help='Disable categorical encoding')
    train_parser.add_argument('--impute', choices=['mean','median','most_frequent','constant'], default=None,
                              help='Imputer strategy for numeric features')
    train_parser.add_argument('--no-scale', action='store_true', help='Disable numeric scaling')
    # Baseline option
    train_parser.add_argument('--baseline', action='store_true', help='Skip TPOT and train baseline model')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict with a trained model')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--data', required=True, help='Path to prediction data (CSV)')
    predict_parser.add_argument('--output', default='predictions.csv', help='Where to save predictions CSV')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--data', required=True, help='Evaluation CSV')
    eval_parser.add_argument('--target', required=True, help='Target column')
    eval_parser.add_argument('--output-dir', default='evaluation', help='Directory for evaluation artifacts')

    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Standalone data visualizations')
    viz_parser.add_argument('--data', required=True, help='CSV path')
    viz_parser.add_argument('--target', required=True, help='Target column')
    viz_parser.add_argument('--output-dir', default='visualizations', help='Directory for plots')

    # Version command
    subparsers.add_parser('version', help='Show version info')
    
    return parser

def train_model(args):
    """Execute model training command."""
    logger.info(f"Starting training task={args.task}")
    overlay_path = None
    try:
        from core import run_automl, _load_config
        from visualization import create_evaluation_dashboard
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Determine task from config default if not provided
        cfg = _load_config(args.config)
        effective_task = args.task or cfg.get('default_task', 'classification')

        # Handle preprocessing overrides
        overrides = {}
        if args.no_categoricals:
            overrides['handle_categoricals'] = False
        if args.impute is not None:
            overrides['impute_strategy'] = args.impute
        if args.no_scale:
            overrides['scale_numeric'] = False

        config_path = args.config
        if overrides and not config_path:
            import tempfile
            import os as temp_os
            import yaml
            fd, overlay_path = tempfile.mkstemp(prefix='evo_cfg_', suffix='.yaml')
            temp_os.close(fd)
            Path(overlay_path).write_text(yaml.safe_dump(overrides))
            config_path = overlay_path

        # Run AutoML
        result = run_automl(
            data_path=args.data,
            target_column=args.target,
            task=effective_task,
            generations=args.generations,
            population_size=args.population,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir,
            max_time_mins=args.max_time,
            max_eval_time_mins=args.max_eval_time,
            config_path=config_path,
            always_baseline=args.baseline
        )

        # Display results
        print("\nTraining Results:\n=================")
        metrics = result.get('metrics', {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        print(f"\nModel saved to: {result['model_path']}")
        if result.get('pipeline_path'):
            print(f"Pipeline code exported to: {result.get('pipeline_path')}")

        # Generate visualizations if requested
        if args.visualize:
            out_dir = output_path / 'visualizations'
            out_dir.mkdir(parents=True, exist_ok=True)
            create_evaluation_dashboard(results=result, output_dir=out_dir)
            logger.info(f"Visualizations saved to {out_dir}")

        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    finally:
        # Clean up temporary config file
        if overlay_path:
            try:
                overlay_file = Path(overlay_path)
                if overlay_file.exists():
                    overlay_file.unlink()
            except Exception:
                pass

def predict_with_model(args):
    """Execute model prediction command."""
    logger.info(f"Predicting with model={args.model}")
    try:
        from core import load_model, load_data, predict
        
        model = load_model(args.model)
        data = load_data(args.data)
        predictions = predict(model, data)
        
        # Save predictions
        pd.DataFrame(predictions, columns=['prediction']).to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1

def evaluate_model(args):
    """Execute model evaluation command."""
    logger.info(f"Evaluating model={args.model}")
    try:
        from core import load_model, load_data
        from utils import get_metrics
        from visualization import (
            create_evaluation_dashboard,
            save_roc_curve,
            save_pr_curve,
            save_residuals,
            save_actual_vs_pred
        )
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load model and data
        model = load_model(args.model)
        data = load_data(args.data)
        X = data.drop(columns=[args.target])
        y = data[args.target]
        y_pred = model.predict(X)

        # Get probabilities for binary classification
        y_proba = None
        if hasattr(model, 'predict_proba') and y.nunique() == 2:
            try:
                y_proba = model.predict_proba(X)[:, 1]
            except Exception:
                y_proba = None
                
        # Calculate metrics
        metrics = get_metrics(y, y_pred, y_proba)

        # Save evaluation report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = output_path / f"evaluation_report_{timestamp}.json"
        
        report_file.write_text(json.dumps({
            'metrics': metrics,
            'y_true': y.tolist(),
            'y_pred': y_pred.tolist(),
            'y_proba': (y_proba.tolist() if y_proba is not None else None)
        }, indent=4))
            
        # Generate plots
        if y_proba is not None:
            save_roc_curve(y, y_proba, output_path / f"roc_{timestamp}.png")
            save_pr_curve(y, y_proba, output_path / f"pr_{timestamp}.png")
            
        if np.issubdtype(y.dtype, np.number):
            save_residuals(y, y_pred, output_path / f"residuals_{timestamp}.png")
            save_actual_vs_pred(y, y_pred, output_path / f"actual_vs_pred_{timestamp}.png")

        # Create dashboard
        create_evaluation_dashboard(
            {'metrics': metrics, 'y_true': y.tolist(), 'y_pred': y_pred.tolist()}, 
            output_path
        )
        
        print(f"Evaluation report saved to: {report_file}")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

def create_visualizations(args):
    """Execute visualization creation command."""
    logger.info(f"Visualizing data={args.data}")
    try:
        from core import load_data
        from visualization import plot_feature_distributions, plot_correlation_matrix
        
        output_path = Path(args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        data = load_data(args.data)
        plot_feature_distributions(data=data, target_column=args.target, output_dir=output_path)
        plot_correlation_matrix(data=data, output_path=output_path / 'correlation_matrix.png')
        
        print(f"Visualizations saved to: {args.output_dir}")
        return 0
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1

def show_version(args=None):
    """Display version information."""
    print("\nEvo-Learn AutoML Toolkit v1.2.0")
    print("Professional automated machine learning with TPOT")
    print("Features: Robust preprocessing, baseline fallbacks, comprehensive evaluation")
    print("Python: 3.14+ compatible with modern features")
    
    try:
        import sys
        import tpot, sklearn, pandas, numpy
        print(f"\nPython version: {sys.version}")
        print("\nDependencies:")
        print(f"- TPOT: {tpot.__version__}")
        print(f"- scikit-learn: {sklearn.__version__}")
        print(f"- pandas: {pandas.__version__}")
        print(f"- numpy: {numpy.__version__}")
    except Exception:
        pass
        
    return 0

def main():
    """Main CLI entry point using pattern matching."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
        
    # Route to appropriate command handler using match statement
    match args.command:
        case 'train':
            return train_model(args)
        case 'predict':
            return predict_with_model(args)
        case 'evaluate':
            return evaluate_model(args)
        case 'visualize':
            return create_visualizations(args)
        case 'version':
            return show_version(args)
        case _:
            parser.print_help()
            return 1

if __name__ == "__main__":
    sys.exit(main())
