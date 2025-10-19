#!/usr/bin/env python3
"""Lightweight smoke test for Evo-Learn core functionality.
Generates a tiny synthetic dataset and ensures run_automl returns metrics.
"""
from enhanced_core import run_automl
import pandas as pd

if __name__ == "__main__":
    df = pd.DataFrame({
        'f1':[0,1,0,1,0,1,0,1,0,1],
        'f2':[1,1,0,0,1,1,0,0,1,0],
        'target':[0,1,0,1,0,1,0,1,0,1]
    })
    df.to_csv('tmp.csv', index=False)
    res = run_automl('tmp.csv', 'target', task='classification', generations=2, population_size=10, max_eval_time_mins=2)
    print(res['metrics'])
