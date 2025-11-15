# Installation Guide

This guide covers different methods to install and set up Evo-Learn.

## Table of Contents

- [Quick Install](#quick-install)
- [From Source](#from-source)
- [Docker Installation](#docker-installation)
- [Development Setup](#development-setup)
- [Troubleshooting](#troubleshooting)

## Quick Install

### PyPI (Coming Soon)

Once published to PyPI:

```bash
pip install evo-learn
```

### From GitHub (Current)

```bash
pip install git+https://github.com/TurboRx/Evo-Learn.git
```

## From Source

### Prerequisites

- Python 3.10 or 3.11 (required for TPOT compatibility)
- pip (latest version recommended)
- Virtual environment tool (venv or conda)
- Git

### Step-by-Step Installation

1. **Clone the repository**

```bash
git clone https://github.com/TurboRx/Evo-Learn.git
cd Evo-Learn
```

2. **Create virtual environment**

Using venv:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

Using conda:
```bash
conda create -n evo-learn python=3.11
conda activate evo-learn
```

3. **Upgrade pip**

```bash
pip install --upgrade pip
```

4. **Install dependencies**

For basic usage:
```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -e ".[dev]"
```

5. **Verify installation**

```bash
python check_installation.py
```

Expected output:
```
✅ All core modules imported successfully
✅ TPOT version: 0.12.x
✅ scikit-learn version: 1.4.x
✅ Environment verified
```

## Docker Installation

### Using Docker Compose (Recommended)

1. **Clone the repository**

```bash
git clone https://github.com/TurboRx/Evo-Learn.git
cd Evo-Learn
```

2. **Build and run**

```bash
# Build image
docker-compose build

# Run CLI
docker-compose run evo-learn python cli.py --help

# Interactive development
docker-compose run evo-learn-dev bash

# Run tests
docker-compose run evo-learn-test
```

### Using Docker Directly

```bash
# Build image
docker build -t evo-learn .

# Run container
docker run -v $(pwd)/data:/app/data evo-learn python cli.py train --data /app/data/your_dataset.csv --target target
```

### Docker Tips

- Mount local directories for data and models:
  ```bash
  docker run -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models evo-learn
  ```

- Use GPU acceleration (if available):
  ```bash
  docker run --gpus all evo-learn
  ```

## Development Setup

### Full Development Environment

1. **Install with development dependencies**

```bash
pip install -e ".[dev]"
```

2. **Install pre-commit hooks**

```bash
pre-commit install
```

3. **Verify setup**

```bash
# Run all tests
pytest tests/ -v

# Check code style
black --check .
flake8 .
isort --check-only .

# Type checking
mypy evo_learn/
```

### IDE Configuration

#### VS Code

Create `.vscode/settings.json`:

```json
{
  "python.linting.enabled": true,
  "python.linting.flake8Enabled": true,
  "python.formatting.provider": "black",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests/"],
  "editor.formatOnSave": true
}
```

#### PyCharm

1. Go to Settings → Tools → Black
2. Enable "On code reformat"
3. Enable "On save"
4. Configure pytest as test runner

## Troubleshooting

### Common Issues

#### Python Version Error

**Problem**: TPOT fails with Python 3.12+

**Solution**: Use Python 3.10 or 3.11
```bash
pyenv install 3.11.0
pyenv local 3.11.0
```

#### Dependency Conflicts

**Problem**: Conflicting package versions

**Solution**: Use fresh virtual environment
```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'evo_learn'`

**Solution**: Install in editable mode
```bash
pip install -e .
```

#### SHAP Installation Issues

**Problem**: SHAP fails to install on some systems

**Solution**: Install build dependencies first

On Ubuntu/Debian:
```bash
sudo apt-get install build-essential
```

On macOS:
```bash
xcode-select --install
```

On Windows:
- Install Visual C++ Build Tools
- Or use conda: `conda install -c conda-forge shap`

#### Memory Errors

**Problem**: Out of memory during TPOT optimization

**Solution**: Reduce parameters in config
```yaml
generations: 3
population_size: 10
max_time_mins: 5
```

#### Slow Performance

**Problem**: Training takes too long

**Solutions**:
1. Use baseline models: `--baseline`
2. Reduce TPOT parameters
3. Enable multiprocessing (use with caution):
   ```python
   tpot = TPOTClassifier(n_jobs=-1)
   ```

### Platform-Specific Notes

#### Windows

- Use `python -m pip` instead of `pip`
- Activate venv: `.venv\Scripts\activate`
- Some packages may need Visual C++ Build Tools

#### macOS (M1/M2)

- Use Python 3.11 for best compatibility
- Install via Homebrew: `brew install python@3.11`
- Some scientific packages benefit from conda

#### Linux

- Install system dependencies:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev build-essential
  ```

### Getting Help

If you encounter issues:

1. Check [existing issues](https://github.com/TurboRx/Evo-Learn/issues)
2. Review [troubleshooting guide](../README.md#troubleshooting)
3. [Open a new issue](https://github.com/TurboRx/Evo-Learn/issues/new) with:
   - Python version: `python --version`
   - Package versions: `pip list | grep -E "tpot|sklearn|pandas"`
   - Operating system
   - Full error message

## Next Steps

After installation:

- Read the [Quick Start Guide](../README.md#quick-start)
- Explore [Examples](../examples/)
- Review [Contributing Guidelines](../CONTRIBUTING.md)
- Check [API Documentation](./API.md) (coming soon)

## Verification Checklist

- [ ] Python 3.10, 3.11, or 3.12 installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] `python check_installation.py` passes
- [ ] Can import: `from core import run_automl`
- [ ] Tests run successfully: `pytest tests/`
- [ ] Pre-commit hooks working (dev only)

---

**Need help?** Open an issue on [GitHub](https://github.com/TurboRx/Evo-Learn/issues)
