#!/usr/bin/env python3
"""
Command Line Interface for Evo-Learn

This script provides a user-friendly command-line interface for using 
the Evo-Learn automated machine learning tool.
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
    train_parser.add_argument('--output-dir', default='models', 
                             help='Directory to save models and results')
    train_parser.add_argument('--max-time', type=int, default=None, 
                             help='Maximum time in minutes for optimization')
    train_parser.add_argument('--random-state', type=int, default=42, 
                             help='Random seed for reproducibility')
    train_parser.add_argument('--visualize', action='store_true', 
                             help='Generate visualizations')
    
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
        # Dynamically import here to avoid overhead when running other commands
        from mloptimizer.core import run_automl
        
        # Create output directory if it doesn't exist
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run AutoML
        result = run_automl(
            data_path=args.data,
            target_column=args.target,
            task=args.task,
            generations=args.generations,
            population_size=args.population,
            test_size=args.test_size,
            random_state=args.random_state,
            output_dir=args.output_dir,
            max_time_mins=args.max_time
        )
        
        logger.info(f"Model training completed successfully")
        
        # Display metrics
        print("\nTraining Results:")
        print("================")
        
        if args.task == 'classification':
            print(f"Accuracy: {result['metrics']['accuracy']:.4f}")
            print(f"F1 Score: {result['metrics']['f1_score']:.4f}")
            if 'roc_auc' in result['metrics']:
                print(f"ROC AUC: {result['metrics']['roc_auc']:.4f}")
        else:  # regression
            print(f"RMSE: {result['metrics']['rmse']:.4f}")
            print(f"R² Score: {result['metrics']['r2']:.4f}")
        
        print(f"\nModel saved to: {result['model_path']}")
        print(f"Pipeline code exported to: {result['pipeline_path']}")
        
        # Generate visualizations if requested
        if args.visualize:
            from mloptimizer.visualization import create_evaluation_dashboard
            
            # Create dashboard based on results
            create_evaluation_dashboard(
                results=result,
                output_dir=os.path.join(args.output_dir, 'visualizations')
            )
            
            logger.info(f"Visualizations saved to {os.path.join(args.output_dir, 'visualizations')}")
        
        return 0
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return 1

def predict_with_model(args):
    """Make predictions with a trained model"""
    logger.info(f"Making predictions with model: {args.model}")
    
    try:
        # Import necessary functions
        from mloptimizer.core import load_model, predict, load_data
        
        # Load the model
        model = load_model(args.model)
        
        # Load the data
        data = load_data(args.data)
        
        # Make predictions
        predictions = predict(model, data)
        
        # Save predictions to file
        pd.DataFrame(predictions, columns=['prediction']).to_csv(args.output, index=False)
        
        logger.info(f"Made predictions for {len(predictions)} samples")
        logger.info(f"Predictions saved to {args.output}")
        
        print(f"\nPredictions saved to: {args.output}")
        print(f"Total predictions: {len(predictions)}")
        
        return 0
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        return 1

def evaluate_model(args):
    """Evaluate a trained model"""
    logger.info(f"Evaluating model: {args.model}")
    
    try:
        # Import necessary functions
        from mloptimizer.core import load_model, load_data
        from mloptimizer.utils import get_metrics
        from mloptimizer.visualization import create_evaluation_dashboard
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load the model and data
        model = load_model(args.model)
        data = load_data(args.data)
        
        # Split data and target
        X = data.drop(args.target, axis=1)
        y = data[args.target]
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probability predictions if available (for classification)
        y_proba = None
        if hasattr(model, 'predict_proba') and len(np.unique(y)) == 2:
            y_proba = model.predict_proba(X)[:, 1]
        
        # Calculate metrics
        metrics = get_metrics(y, y_pred, y_proba)
        
        # Create evaluation report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(args.output_dir, f"evaluation_report_{timestamp}.json")
        
        # Prepare results dictionary
        results = {
            'model_info': {
                'model_path': args.model,
                'evaluation_data': args.data,
                'timestamp': timestamp,
                'task': 'classification' if len(np.unique(y)) <= 10 else 'regression'
            },
            'metrics': metrics,
            'y_true': y.values.tolist(),
            'y_pred': y_pred.tolist(),
            'feature_names': X.columns.tolist()
        }
        
        if y_proba is not None:
            results['y_proba'] = y_proba.tolist()
        
        # Save report
        with open(report_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json.dump(results, f, indent=4)
        
        logger.info(f"Evaluation report saved to {report_file}")
        
        # Create visualizations
        create_evaluation_dashboard(results, args.output_dir)
        
        # Display metrics
        print("\nEvaluation Results:")
        print("==================")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        print(f"\nDetailed evaluation report saved to: {report_file}")
        print(f"Visualizations saved to: {args.output_dir}")
        
        return 0
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return 1

def create_visualizations(args):
    """Create visualizations for data"""
    logger.info(f"Creating visualizations for {args.data}")
    
    try:
        # Import necessary functions
        from mloptimizer.visualization import (
            plot_feature_distributions, 
            plot_correlation_matrix
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Load data
        from mloptimizer.core import load_data
        data = load_data(args.data)
        
        # Create feature distribution plots
        plot_feature_distributions(
            data=data,
            target_column=args.target,
            output_dir=args.output_dir
        )
        
        # Create correlation matrix
        plot_correlation_matrix(
            data=data,
            output_path=os.path.join(args.output_dir, 'correlation_matrix.png')
        )
        
        logger.info(f"Visualizations saved to {args.output_dir}")
        
        print(f"\nVisualizations saved to: {args.output_dir}")
        
        return 0
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}")
        return 1

def show_version():
    """Show Evo-Learn version information"""
    version_info = {
        'name': 'Evo-Learn',
        'version': '1.0.0',
        'description': 'Automated machine learning tool based on TPOT',
        'repository': 'https://github.com/TurboRx/Evo-Learn'
    }
    
    print(f"\n{version_info['name']} v{version_info['version']}")
    print(f"{version_info['description']}")
    print(f"Repository: {version_info['repository']}")
    
    # Try to get dependency versions
    try:
        import tpot
        import sklearn
        import pandas
        import numpy
        
        print("\nDependencies:")
        print(f"- TPOT: {tpot.__version__}")
        print(f"- scikit-learn: {sklearn.__version__}")
        print(f"- pandas: {pandas.__version__}")
        print(f"- numpy: {numpy.__version__}")
    except ImportError:
        pass
    
    return 0

def main():
    """Main entry point for the CLI"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute appropriate function based on command
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