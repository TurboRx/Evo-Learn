#!/usr/bin/env python3
"""
Validation Script for Evo-Learn AutoML Toolkit

This script performs comprehensive validation to ensure that all necessary
dependencies are installed and core functionality is working correctly.
"""

import importlib
import importlib.util
import os
import sys
from typing import List, Tuple


def check_imports(packages: List[str]) -> Tuple[bool, List[str]]:
    """
    Check if the given packages can be imported.
    
    Args:
        packages: List of package names to check
        
    Returns:
        Tuple with success flag and list of missing packages
    """
    missing = []
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ Successfully imported {package}")
        except ImportError as e:
            print(f"✗ Failed to import {package}: {e}")
            missing.append(package)
    
    return len(missing) == 0, missing


def check_core_modules() -> Tuple[bool, List[str]]:
    """
    Check if the core Evo-Learn modules exist and can be imported.
    
    Returns:
        Tuple with success flag and list of missing modules
    """
    core_modules = [
        "core.py",
        "utils.py", 
        "visualization.py",
        "cli.py",
        "preprocessing.py"
    ]
    
    missing = []
    for module in core_modules:
        if not os.path.exists(module):
            print(f"✗ Core module {module} not found")
            missing.append(module)
            continue
            
        module_name = module.replace(".py", "")
        try:
            # Try with direct file import approach which is safer
            spec = importlib.util.spec_from_file_location(module_name, module)
            if spec is not None and spec.loader is not None:
                module_obj = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module_obj)
                print(f"✓ Successfully imported {module}")
            else:
                # Fallback to normal import
                try:
                    importlib.import_module(module_name)
                    print(f"✓ Successfully imported {module}")
                except ImportError:
                    print(f"✗ Could not import {module}")
                    missing.append(module)
        except Exception as e:
            print(f"✗ Failed to import {module}: {e}")
            missing.append(module)
    
    return len(missing) == 0, missing


def check_configuration_files() -> Tuple[bool, List[str]]:
    """
    Check if required configuration files exist.
    
    Returns:
        Tuple with success flag and list of missing files
    """
    config_files = [
        "config.yaml",
        "pyproject.toml", 
        "requirements.txt"
    ]
    
    missing = []
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ Configuration file {config_file} found")
        else:
            print(f"✗ Configuration file {config_file} not found")
            missing.append(config_file)
    
    return len(missing) == 0, missing


def test_basic_functionality():
    """
    Test basic functionality of core modules.
    
    Returns:
        bool: True if basic tests pass, False otherwise
    """
    try:
        # Test core module import
        import core
        print("✓ Core module functionality test passed")
        
        # Test utilities
        import utils
        print("✓ Utils module functionality test passed")
        
        # Test visualization
        import visualization
        print("✓ Visualization module functionality test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Run comprehensive validation checks."""
    print("\n===== Evo-Learn AutoML Toolkit Validation =====")
    print("Performing comprehensive system validation...\n")
    
    # Check required packages
    print("----- Checking Required Dependencies -----")
    packages = [
        "numpy", "pandas", "sklearn", "matplotlib", "seaborn", 
        "tpot", "yaml", "scipy", "joblib"
    ]
    pkg_success, missing_pkgs = check_imports(packages)
    
    # Check core modules
    print("\n----- Checking Core Modules -----")
    mod_success, missing_mods = check_core_modules()
    
    # Check configuration files
    print("\n----- Checking Configuration Files -----")
    config_success, missing_configs = check_configuration_files()
    
    # Test basic functionality
    print("\n----- Testing Basic Functionality -----")
    func_success = test_basic_functionality()
    
    # Print detailed summary
    print("\n----- Validation Summary -----")
    
    if pkg_success:
        print("✓ All required dependencies are installed")
    else:
        print(f"✗ Missing dependencies: {', '.join(missing_pkgs)}")
        print("  Install with: pip install -r requirements.txt")
        
    if mod_success:
        print("✓ All core modules are present and importable")
    else:
        print(f"✗ Issues with modules: {', '.join(missing_mods)}")
        
    if config_success:
        print("✓ All configuration files are present")
    else:
        print(f"✗ Missing configuration files: {', '.join(missing_configs)}")
        
    if func_success:
        print("✓ Basic functionality tests passed")
    else:
        print("✗ Basic functionality tests failed")
    
    # Overall status
    print("\n----- Overall Validation Status -----")
    overall_success = pkg_success and mod_success and config_success and func_success
    
    if overall_success:
        print("✓ VALIDATION PASSED - Evo-Learn is ready to use!")
        print("\nNext steps:")
        print("  - Run: python cli.py train --data your_data.csv --target target_column")
        print("  - Or: python cli.py --help for more options")
        return 0
    else:
        print("✗ VALIDATION FAILED - Please address the issues above")
        print("\nTroubleshooting:")
        print("  - Install dependencies: pip install -r requirements.txt")
        print("  - Check file permissions and paths")
        print("  - Ensure Python 3.10+ is being used")
        return 1


if __name__ == "__main__":
    sys.exit(main())