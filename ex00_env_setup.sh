#!/bin/bash

# Create virtual environment
python3 -m venv ex00

# Activate virtual environment
source ex00/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required libraries
pip install pandas numpy jupyter matplotlib scikit-learn

echo "Environment setup complete. To activate, run 'source ex00/bin/activate'"
