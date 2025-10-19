#!/usr/bin/env python3
"""
Command Line Interface for Evo-Learn (enhanced)

Aligned imports to enhanced_* modules; supports --config, preprocessing toggles, and --baseline.
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
    parser = argparse.ArgumentParser(
        description='Evo-Learn: Automated Machine Learning Tool',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Train
    train_parser = subparsers.add_parser('train', help='Train a model using AutoML')
    train_parser.add_argument('--data', required=True, help='Path to CSV data file')
    train_parser.add_argument('--target', required=True, help='Name of target column')
    train_parser.add_argument('--task', choices=['classification', 'regression'], default=None,
                             help='Machine learning task (defaults to config default_task or classification)')
    train_parser.add_argument('--generations', type=int, default=5, help='TPOT generations')
    train_parser.add_argument('--population', type=int, default=20, help='TPOT population size')
    train_parser.add_argument('--test-size', type=float, default=0.2, help='Test split size')
    train_parser.add_argument('--output-dir', default='mloptimizer/models', help='Model output directory')
    train_parser.add_argument('--max-time', type=int, default=None, help='Total TPOT time (mins)')
    train_parser.add_argument('--max-eval-time', type=int, default=5, help='Per-pipeline eval time (mins)')
    train_parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    train_parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    # Config & preprocessing
    train_parser.add_argument('--config', help='Path to evo_config.yaml')
    train_parser.add_argument('--no-categoricals', action='store_true', help='Disable categorical encoding')
    train_parser.add_argument('--impute', choices=['mean','median','most_frequent','constant'], default=None,
                              help='Imputer strategy for numeric features')
    train_parser.add_argument('--no-scale', action='store_true', help='Disable numeric scaling')
    # Baseline
    train_parser.add_argument('--baseline', action='store_true', help='Skip TPOT and train baseline model')

    # Predict
    predict_parser = subparsers.add_parser('predict', help='Predict with a trained model')
    predict_parser.add_argument('--model', required=True, help='Path to trained model')
    predict_parser.add_argument('--data', required=True, help='Path to prediction data (CSV)')
    predict_parser.add_argument('--output', default='predictions.csv', help='Where to save predictions CSV')

    # Evaluate
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model', required=True, help='Path to trained model')
    eval_parser.add_argument('--data', required=True, help='Evaluation CSV')
    eval_parser.add_argument('--target', required=True, help='Target column')
    eval_parser.add_argument('--output-dir', default='evaluation', help='Directory for evaluation artifacts')

    # Visualize
    viz_parser = subparsers.add_parser('visualize', help='Standalone data visualizations')
    viz_parser.add_argument('--data', required=True, help='CSV path')
    viz_parser.add_argument('--target', required=True, help='Target column')
    viz_parser.add_argument('--output-dir', default='visualizations', help='Directory for plots')

    subparsers.add_parser('version', help='Show version info')
    return parser

def train_model(args):
    logger.info(f"Starting training task={args.task}")
    try:
        from enhanced_core import run_automl, _load_config
        from enhanced_visualization import create_evaluation_dashboard
        os.makedirs(args.output_dir, exist_ok=True)

        # Determine task from config default if not provided
        cfg = _load_config(args.config)
        effective_task = args.task or cfg.get('default_task', 'classification')

        overrides = {}
        if args.no_categoricals:
            overrides['handle_categoricals'] = False
        if args.impute is not None:
            overrides['impute_strategy'] = args.impute
        if args.no_scale:
            overrides['scale_numeric'] = False

        config_path = args.config
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

        print("\nTraining Results:\n================")
        metrics = result.get('metrics', {})
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
        print(f"\nModel saved to: {result['model_path']}")
        print(f"Pipeline code exported to: {result.get('pipeline_path')}")

        if args.visualize:
            out_dir = os.path.join(args.output_dir, 'visualizations')
            os.makedirs(out_dir, exist_ok=True)
            create_evaluation_dashboard(results=result, output_dir=out_dir)
            logger.info(f"Visualizations saved to {out_dir}")

        if overlay_path:
            try:
                os.remove(overlay_path)
            except Exception:
                pass
        return 0
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1

def predict_with_model(args):
    logger.info(f"Predicting with model={args.model}")
    try:
        from enhanced_core import load_model, load_data, predict
        model = load_model(args.model)
        data = load_data(args.data)
        preds = predict(model, data)
        pd.DataFrame(preds, columns=['prediction']).to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
        return 0
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1

def evaluate_model(args):
    logger.info(f"Evaluating model={args.model}")
    try:
        from enhanced_core import load_model, load_data
        from enhanced_utils import get_metrics
        from enhanced_visualization import (
            create_evaluation_dashboard,
            save_roc_curve,
            save_pr_curve,
            save_residuals,
            save_actual_vs_pred
        )
        os.makedirs(args.output_dir, exist_ok=True)
        model = load_model(args.model)
        data = load_data(args.data)
        X = data.drop(columns=[args.target])
        y = data[args.target]
        y_pred = model.predict(X)

        # Proba if binary classification
        y_proba = None
        if hasattr(model, 'predict_proba') and y.nunique() == 2:
            try:
                y_proba = model.predict_proba(X)[:, 1]
            except Exception:
                y_proba = None
        metrics = get_metrics(y, y_pred, y_proba)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(args.output_dir, f"evaluation_report_{ts}.json")
        with open(report_file, 'w') as f:
            json.dump({
                'metrics': metrics,
                'y_true': y.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': (y_proba.tolist() if y_proba is not None else None)
            }, f, indent=4)
        # Plots
        if y_proba is not None:
            save_roc_curve(y, y_proba, os.path.join(args.output_dir, f"roc_{ts}.png"))
            save_pr_curve(y, y_proba, os.path.join(args.output_dir, f"pr_{ts}.png"))
        if np.issubdtype(y.dtype, np.number):
            save_residuals(y, y_pred, os.path.join(args.output_dir, f"residuals_{ts}.png"))
            save_actual_vs_pred(y, y_pred, os.path.join(args.output_dir, f"actual_vs_pred_{ts}.png"))

        create_evaluation_dashboard({'metrics': metrics, 'y_true': y.tolist(), 'y_pred': y_pred.tolist()}, args.output_dir)
        print(f"Evaluation report saved to: {report_file}")
        return 0
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1

def create_visualizations(args):
    logger.info(f"Visualizing data={args.data}")
    try:
        from enhanced_core import load_data
        from enhanced_visualization import (
            plot_feature_distributions,
            plot_correlation_matrix
        )
        os.makedirs(args.output_dir, exist_ok=True)
        data = load_data(args.data)
        plot_feature_distributions(data=data, target_column=args.target, output_dir=args.output_dir)
        plot_correlation_matrix(data=data, output_path=os.path.join(args.output_dir, 'correlation_matrix.png'))
        print(f"Visualizations saved to: {args.output_dir}")
        return 0
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        return 1

def show_version():
    print("\nEvo-Learn (enhanced) v1.0.0")
    print("Automated ML with TPOT + robust preprocessing + baseline fallback")
    try:
        import tpot, sklearn, pandas, numpy
        print("\nDependencies:")
        print(f"- TPOT: {tpot.__version__}")
        print(f"- scikit-learn: {sklearn.__version__}")
        print(f"- pandas: {pandas.__version__}")
        print(f"- numpy: {numpy.__version__}")
    except Exception:
        pass
    return 0

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    if args.command == 'train':
        return train_model(args)
    if args.command == 'predict':
        return predict_with_model(args)
    if args.command == 'evaluate':
        return evaluate_model(args)
    if args.command == 'visualize':
        return create_visualizations(args)
    if args.command == 'version':
        return show_version()
    parser.print_help()
    return 1

if __name__ == "__main__":
    sys.exit(main())
