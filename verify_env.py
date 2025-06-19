#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ML Environment Verification Script

This script verifies that the ml-env conda environment is properly set up
and all required libraries are installed.
"""

import sys
import platform
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def print_environment_info():
    """Print information about the Python environment."""
    print("\n=== Python Environment Information ===")
    print(f"Python version: {platform.python_version()}")
    print(f"Python executable path: {sys.executable}")
    print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'Not in a conda environment')}")
    print(f"Conda prefix: {os.environ.get('CONDA_PREFIX', 'No conda prefix found')}")
    print("=====================================\n")

def check_libraries():
    """Check if required libraries are installed."""
    print("=== Library Versions ===")
    libraries = {
        'numpy': np,
        'pandas': pd,
        'matplotlib': plt,
        'seaborn': sns,
        'sklearn': datasets
    }
    
    for name, lib in libraries.items():
        version = getattr(lib, '__version__', 'unknown version')
        print(f"{name}: {version}")
    print("=======================\n")

def run_simple_ml():
    """Run a simple machine learning example."""
    print("=== Running Simple ML Example ===")
    
    # Load iris dataset
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    print("==============================\n")

def main():
    """Main function."""
    print("\nML Environment Verification Script")
    print("================================")
    
    print_environment_info()
    check_libraries()
    run_simple_ml()
    
    print("Environment verification complete! Everything is working correctly.")

if __name__ == "__main__":
    main()
