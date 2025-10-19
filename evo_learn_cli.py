#!/usr/bin/env python3
"""
Command Line Interface for Evo-Learn (enhanced)

Adds --config support and preprocessing toggles passed through to run_automl.
"""

import argparse
import logging
import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

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
    """Set up argument parser for command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evo-Learn: Automated Machine Learning Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model using AutoML')
    train_parser.add_argument('--data', required=True, help='Path to CSV data file')
    train_parser.add_argument('--target', required=True, help='Name of target column')
    train_parser.add_argument('--task', choices=['classification', 'regression'], 
                             default='classification', help='Machine learning task')
    train_parser.add_argument('--generations', type=int, default=5, 
                             help='Number of generations for TPOT')
    train_parser.add_argument('--population', type=int, default=20, 
                             help='Population size for TPOT')
    train_parser.add_argument('--test-size', type=float, default=0.2, 
                             help='Proportion of data for testing')
    train_parser.add_argument('--output-dir', default='mloptimizer/models', 
                             help='Directory to save models and results')
    train_parser.add_argument('--max-time', type=int, default=None, 
                             help='Maximum time in minutes for optimization')
    train_parser.add_argument('--max-eval-time', type=int, default=5,
                             help='Maximum time per pipeline evaluation in minutes')
    train_parser.add_argument('--random-state', type=int, default=42, 
                             help='Random seed for reproducibility')
    train_parser.add_argument('--visualize', action='store_true', 
                             help='Generate visualizations')
    # Config and preprocessing toggles
    train_parser.add_argument('--config', help='Path to evo_config.yaml for overrides')
    train_parser.add_argument('--no-categoricals', action='store_true', help='Disable automatic categorical encoding')
    train_parser.add_argument('--impute', choices=['mean','median','most_frequent','constant'], default=None,
                              help='Imputation strategy override for numeric features')
    train_parser.add_argument('--no-scale', action='store_true', help='Disable numeric scaling')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Make predictions with a trained model')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--data', required=True, help='Path to data for prediction')
    predict_parser.add_argument('--output', default='predictions.csv', 
                              help='File to save predictions')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--data', required=True, help='Path to evaluation data')
    eval_parser.add_argument('--target', required=True, help='Name of target column')
    eval_parser.add_argument('--output-dir', default='evaluation', 
                           help='Directory to save evaluation results')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Create visualizations for data')
    viz_parser.add_argument('--data', required=True, help='Path to CSV data file')
    viz_parser.add_argument('--target', required=True, help='Name of target column')
    viz_parser.add_argument('--output-dir', default='visualizations', 
                          help='Directory to save visualizations')
    
    # Version command
    subparsers.add_parser('version', help='Show Evo-Learn version information')
    
    return parser

def train_model(args):
    """Train a model using AutoML"""
    logger.info(f"Starting {args.task} model training with Evo-Learn")
    
    try:
        from enhanced_core import run_automl
        
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Build overrides from CLI flags
        config_path = args.config
        # Preprocessing overrides (only if provided)
        overrides = {}
        if args.no_categoricals:
            overrides['handle_categoricals'] = False
        if args.impute is not None:
            overrides['impute_strategy'] = args.impute
        if args.no_scale:
            overrides['scale_numeric'] = False
        
        # If overrides exist but no config path provided, write a temp overlay
        overlay_path = None
        if overrides and not config_path:
            import tempfile, yaml
            fd, overlay_path = tempfile.mkstemp(prefix='evo_cfg_', suffix='.yaml')
            os.close(fd)
            with open(overlay_path, 'w') as f:
                yaml.safe_dump(overrides, f)
            config_path = overlay_path
        
        result = run_automl(
            data_path=args.data,
            target_column=args.target,
            task=args.task,
            generations=args.generations,
            population_size=args.population,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir,
            max_time_mins=args.max_time,
            max_eval_time_mins=args.max_eval_time,
            config_path=config_path
        )
        
        logger.info(f"Model training completed successfully")
        
        print("\nTraining Results:")
        print("================")
        if args.task == 'classification':
            print(f"Accuracy: {result['metrics'].get('accuracy', float('nan')):.4f}")
            if 'f1_score' in result['metrics']:
                print(f"F1 Score: {result['metrics']['f1_score']:.4f}")
            if 'roc_auc' in result['metrics']:
                print(f"ROC AUC: {result['metrics']['roc_auc']:.4f}")
        else:
            if 'rmse' in result['metrics']:
                print(f"RMSE: {result['metrics']['rmse']:.4f}")
            if 'r2' in result['metrics']:
                print(f"R² Score: {result['metrics']['r2']:.4f}")
        
        print(f"\nModel saved to: {result['model_path']}")
        print(f"Pipeline code exported to: {result.get('pipeline_path')}")
        
        if overlay_path:
            try:
                os.remove(overlay_path)
            except Exception:
                pass
        
        return 0
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return 1

# predict_with_model, evaluate_model, create_visualizations, show_version remain unchanged
# (keeping original implementations)
from evo_learn_cli import predict_with_model, evaluate_model, create_visualizations, show_version  # type: ignore

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    if args.command == 'train':
        return train_model(args)
    elif args.command == 'predict':
        return predict_with_model(args)
    elif args.command == 'evaluate':
        return evaluate_model(args)
    elif args.command == 'visualize':
        return create_visualizations(args)
    elif args.command == 'version':
        return show_version()
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())
