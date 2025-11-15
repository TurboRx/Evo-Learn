#!/usr/bin/env python3
"""
Installation Check Script for Evo-Learn

This script performs basic verification to ensure that the necessary
dependencies are installed and core functionality is working correctly.
"""

import importlib
import importlib.util  # Add explicit import for importlib.util
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
        "preprocessing.py",
        "validate.py"
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


def main():
    """Run installation checks"""
    print("\n===== Evo-Learn Installation Check =====\n")
    
    # Check required packages
    print("\n----- Checking Required Packages -----")
    packages = ["numpy", "pandas", "sklearn", "matplotlib", "seaborn", "tpot"]
    pkg_success, missing_pkgs = check_imports(packages)
    
    # Check core modules
    print("\n----- Checking Core Modules -----")
    mod_success, missing_mods = check_core_modules()
    
    # Print summary
    print("\n----- Verification Summary -----")
    if pkg_success:
        print("✓ All required packages are installed")
    else:
        print(f"✗ Missing packages: {', '.join(missing_pkgs)}")
        
    if mod_success:
        print("✓ All core modules are present and importable")
    else:
        print(f"✗ Issues with modules: {', '.join(missing_mods)}")
    
    # Overall status
    print("\n----- Overall Status -----")
    if pkg_success and mod_success:
        print("✓ Verification PASSED")
        return 0
    else:
        print("✗ Verification FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())