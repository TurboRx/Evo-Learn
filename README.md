# Evo-Learn Enhancement Project

This project consists of two main components:

1. **Evo-Learn Enhancements** - A set of improved files and new functionality for the Evo-Learn automated machine learning tool.
2. **Verification Web Interface** - A web application to verify, test, and showcase the Evo-Learn repository.

## Evo-Learn Enhancements

The enhanced files provide significant improvements to the original Evo-Learn repository:

- **Enhanced Core Functionality**
  - Extended automated machine learning capabilities
  - Improved model selection and evaluation
  - Better support for classification and regression tasks
  - Advanced hyperparameter tuning

- **Enhanced Utilities**
  - Advanced timing and performance tracking
  - Improved error handling and validation
  - Cross-validation utilities
  - Model metadata management

- **Enhanced Visualization**
  - Feature distribution visualization
  - Correlation matrix heatmaps
  - Confusion matrices and ROC curves
  - Learning curve analysis
  - Feature importance plots

- **Command Line Interface**
  - User-friendly command-line tool
  - Easy model training, evaluation, and prediction
  - Visualization creation
  - Comprehensive help and documentation

- **Improved Documentation**
  - Enhanced README.md with detailed usage instructions
  - API documentation for all functions
  - Code examples for common tasks
  - Installation and requirements information

## GitHub Update Tool

The `github_update.py` script enables automated updating of the Evo-Learn repository via the GitHub API. It can:

- Create new branches
- Update multiple files
- Commit changes
- Create pull requests

Usage:
```bash
python github_update.py --token YOUR_GITHUB_TOKEN --owner OWNER_NAME --repo REPO_NAME --files enhanced_core.py enhanced_utils.py enhanced_visualization.py evo_learn_cli.py enhanced_README.md
```

## Quick Verification

You can quickly verify your Evo-Learn installation using the included verification script:

```bash
python verify_evo_learn.py
```

This script checks:
- Required package dependencies
- Core module availability
- Module imports

If successful, you'll see a "Verification PASSED" message indicating your environment is properly set up.

## Verification Web Interface

The web interface allows users to:

1. Verify the Evo-Learn repository
2. Test core functionality
3. View verification results
4. Access documentation for enhanced features

### Running the Web Interface

```bash
python app.py
```

Or with Gunicorn:
```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

## Project Structure

```
├── app.py                    # Flask application
├── main.py                   # Entry point for Gunicorn
├── verify_evo_learn.py       # Verification logic
├── github_update.py          # GitHub updating tool
├── enhanced_core.py          # Enhanced core functionality
├── enhanced_utils.py         # Enhanced utility functions
├── enhanced_visualization.py # Enhanced visualization tools
├── evo_learn_cli.py          # Command-line interface
├── enhanced_README.md        # Enhanced documentation
├── templates/                # HTML templates
├── static/                   # Static assets
└── test_evolearn.py          # Test script
```

## License

MIT License

Copyright (c) 2025 Evo-Learn Contributors