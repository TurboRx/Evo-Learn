#!/usr/bin/env python3
"""Lightweight smoke test for Evo-Learn core functionality.
Generates a tiny synthetic dataset and ensures run_automl returns metrics.
"""
import sys
import os
# Add parent directory to path for standalone execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import run_automl
import pandas as pd
import tempfile
import os

if __name__ == "__main__":
    df = pd.DataFrame({
        'f1':[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
        'f2':[1,1,0,0,1,1,0,0,1,0,1,1,0,0,1,1,0,0,1,0],
        'target':[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]
    })
    
    # Use temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    try:
        res = run_automl(
            temp_file.name, 
            'target', 
            task='classification', 
            generations=2, 
            population_size=10, 
            max_eval_time_mins=2,
            always_baseline=True  # Use baseline for faster smoke test
        )
        print("Smoke test passed!")
        print("Metrics:", res['metrics'])
    finally:
        # Clean up
        if os.path.exists(temp_file.name):
            os.unlink(temp_file.name)

