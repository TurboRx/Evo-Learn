# Evo-Learn

A robust AutoML toolkit built on TPOT with production-friendly preprocessing, config-driven runs, baseline fallbacks, and CI-backed verification.

## 📚 Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Config file](#config-file)
- [CLI usage](#cli-usage)
- [Artifacts](#artifacts)
- [CI status](#ci-status)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Author](#author)

## Features
- Automated model search with TPOT (classification and regression)
- Full preprocessing pipeline (imputation, categorical encoding, optional scaling)
- Baseline fallback (LogisticRegression/Ridge) when TPOT fails or when forced
- YAML config overrides for reproducible runs
- Stratified splits for classification by default
- Evaluation helpers and plots (ROC/PR/residuals/actual-vs-pred)
- Clean, pinned requirements for Python 3.10/3.11
- GitHub Actions CI with verification and smoke tests

## Installation

Use Python 3.10 or 3.11 and install pinned deps:

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Verify environment and modules:

```bash
python verify_evo_learn.py
```

## Quick start

Python API:

```python
from enhanced_core import run_automl

res = run_automl(
    data_path="your_dataset.csv",
    target_column="target",
    task="classification",   # or "regression"
    generations=5,
    population_size=20,
    config_path="evo_config.yaml"  # optional
)
print(res["metrics"])  # e.g., accuracy/f1 for classification, rmse/r2 for regression
```

CLI (TPOT mode):

```bash
python evo_learn_cli.py train --data your.csv --target target --config evo_config.yaml
```

Baseline (skip TPOT):

```bash
# via CLI flag
python evo_learn_cli.py train --data your.csv --target target --baseline

# or via config (set baseline: true in evo_config.yaml)
python evo_learn_cli.py train --data your.csv --target target --config evo_config.yaml
```

Predictions:

```bash
python evo_learn_cli.py predict --model path/to/model.pkl --data new_data.csv --output predictions.csv
```

Evaluation with plots:

```bash
python evo_learn_cli.py evaluate --model path/to/model.pkl --data eval.csv --target target --output-dir evaluation
```

## Config file

All defaults can be controlled via evo_config.yaml:

```yaml
default_task: classification
random_state: 42
test_size: 0.2
output_dir: mloptimizer/models
generations: 5
population_size: 20
max_time_mins: null
max_eval_time_mins: 5
handle_categoricals: true
impute_strategy: median
scale_numeric: true
baseline: false
```

CLI flags override config only when provided (e.g., --baseline, --impute, --no-scale).

## Artifacts

Training creates these in output_dir:
- model.pkl: full Pipeline (preprocessing + model)
- *_metadata.json: run metadata, metrics, and settings
- *_pipeline.py: TPOT-exported code when available
- Optional feature_importance.png (if the fitted model exposes importances)

Evaluation saves:
- evaluation_report_*.json: metrics and arrays
- roc_*.png, pr_*.png (binary classification with proba)
- residuals_*.png, actual_vs_pred_*.png (regression)

## CI status

This repo includes a single workflow at .github/workflows/ci.yml that:
- Installs pinned requirements
- Runs verification
- Runs baseline smoke tests (classification + regression)

## Troubleshooting
- Use Python 3.10/3.11 to avoid TPOT issues on newer interpreters
- If TPOT fails or times out, use --baseline or set baseline: true in config
- Mixed data types are supported via automatic imputation/encoding; unknown categories are ignored safely
- For class imbalance, stratified train/test split is applied automatically

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Author

Built by TurboRx
