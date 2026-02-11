#!/bin/bash

# Activate virtual environment
source ex00/bin/activate

echo "Running Exercise 1..."
python3 ex01_estimator.py
echo "-----------------------------------"

echo "Running Exercise 2 (1D Linear Regression)..."
python3 ex02_linear_regression_1d.py
echo "-----------------------------------"

echo "Running Exercise 3 (Train Test Split)..."
python3 ex03_split.py
echo "-----------------------------------"

echo "Running Exercise 4 (Diabetes Forecast)..."
python3 ex04_diabetes.py
echo "-----------------------------------"

echo "Running Exercise 5 (Gradient Descent)..."
python3 ex05_gradient_descent.py
echo "-----------------------------------"

echo "All exercises completed."
