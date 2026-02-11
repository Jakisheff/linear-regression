"""
Audit verification script for all exercises
"""

print("=" * 80)
print("EXERCISE 0: Environment and Libraries")
print("=" * 80)

import sys
print(f"Python version: {sys.version}")
print(f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")

try:
    import jupyter
    import numpy
    import pandas
    import matplotlib
    import sklearn
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")

print("\n" + "=" * 80)
print("EXERCISE 1: Scikit-learn Estimator")
print("=" * 80)

from sklearn.linear_model import LinearRegression
import numpy as np

X = [[1], [2.1], [3]]
y = [[1], [2], [3]]
reg = LinearRegression()
reg.fit(X, y)
x_pred = [[4]]
prediction = reg.predict(x_pred)

print(f"Q1 - Prediction: {prediction}")
print(f"Expected: array([[3.96013289]])")
print(f"Match: {np.allclose(prediction, [[3.96013289]])}")

print(f"\nQ2 - Coefficients: {reg.coef_}")
print(f"     Intercept: {reg.intercept_}")
print(f"     Score: {reg.score(X, y)}")
print(f"Expected Coefficients: [[0.99667774]]")
print(f"Expected Intercept: [-0.02657807]")
print(f"Expected Score: 0.9966777408637874")

print("\n" + "=" * 80)
print("EXERCISE 2: Linear Regression in 1D")
print("=" * 80)

from sklearn.datasets import make_regression

X, y, coef = make_regression(n_samples=100, n_features=1, n_informative=1,
                             noise=10, coef=True, random_state=0, bias=100.0)

reg = LinearRegression()
reg.fit(X, y)

print(f"Q2 - Equation: y = {reg.coef_[0]} * x + {reg.intercept_}")
print(f"Expected: y = 42.619430291366946 * x + 99.18581817296929")
print(f"Match: {np.allclose(reg.coef_[0], 42.619430291366946) and np.allclose(reg.intercept_, 99.18581817296929)}")

y_pred = reg.predict(X)
print(f"\nQ4 - First 10 predictions: {y_pred[:10]}")
expected_pred = np.array([ 83.86186727, 140.80961751, 116.3333897 ,  64.52998689,
        61.34889539, 118.10301628,  57.5347917 , 117.44107847,
       108.06237908,  85.90762675])
print(f"Expected: {expected_pred}")
print(f"Match: {np.allclose(y_pred[:10], expected_pred)}")

from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
print(f"\nQ5 - MSE (noise=10): {mse}")
print(f"Expected: 114.17148616819485")
print(f"Match: {np.allclose(mse, 114.17148616819485)}")

# Now with noise=50
X, y, coef = make_regression(n_samples=100, n_features=1, n_informative=1,
                             noise=50, coef=True, random_state=0, bias=100.0)
reg_50 = LinearRegression()
reg_50.fit(X, y)
y_pred_50 = reg_50.predict(X)
mse_50 = mean_squared_error(y, y_pred_50)
print(f"\nQ6 - MSE (noise=50): {mse_50}")
print(f"Expected: 2854.2871542048706")
print(f"Match: {np.allclose(mse_50, 2854.2871542048706)}")

print("\n" + "=" * 80)
print("EXERCISE 3: Train Test Split")
print("=" * 80)

from sklearn.model_selection import train_test_split

X = np.arange(1, 21).reshape(10, -1)
y = np.arange(1, 11)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

print("Q1 - X_train:")
print(X_train)
print("\ny_train:")
print(y_train)
print("\nX_test:")
print(X_test)
print("\ny_test:")
print(y_test)

print("\n" + "=" * 80)
print("EXERCISE 4: Forecast Diabetes Progression")
print("=" * 80)

from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

print("Q1 - y_train.values[:10]:")
print(y_train.values[:10])
print("Expected: [202.  55. 202.  42. 214. 173. 118.  90. 129. 151.]")

print("\ny_test.values[:10]:")
print(y_test.values[:10])
print("Expected: [ 71.  72. 235. 277. 109.  61. 109.  78.  66. 192.]")

reg = LinearRegression()
reg.fit(X_train, y_train)

print("\nQ2 - Coefficients and Intercept:")
feature_names = list(X.columns) + ['intercept']
coef_values = list(reg.coef_) + [reg.intercept_]
for name, val in zip(feature_names, coef_values):
    print(f"('{name}', {val})")

y_pred = reg.predict(X_test)
print("\nQ3 - predictions_on_test[:10]:")
print(y_pred[:10].reshape(-1, 1))

mse_train = mean_squared_error(y_train, reg.predict(X_train))
mse_test = mean_squared_error(y_test, y_pred)
print(f"\nQ4 - MSE Train: {mse_train}")
print(f"Expected: 2888.326888")
print(f"MSE Test: {mse_test}")
print(f"Expected: 2858.255153")

print("\n" + "=" * 80)
print("EXERCISE 5: Gradient Descent")
print("=" * 80)

X, y, coef = make_regression(n_samples=100, n_features=1, n_informative=1,
                             noise=10, coef=True, random_state=0, bias=100.0)

def compute_mse(coefs, X, y):
    a, b = coefs
    y_preds = a * X.flatten() + b
    mse = np.mean((y_preds - y)**2)
    return mse

print(f"Q2 - MSE for a=1, b=2: {compute_mse([1, 2], X, y)}")
print(f"Expected: 11808.867339751561")

aa, bb = np.mgrid[-200:200:0.5, -200:200:0.5]
grid = np.c_[aa.ravel(), bb.ravel()]

print(f"\nQ3 - grid.shape: {grid.shape}")
print(f"Expected: (640000, 2)")

losses = []
for i in range(len(grid)):
    losses.append(compute_mse(grid[i], X, y))

losses = np.array(losses)
print(f"\nQ4 - First 10 losses:")
print(losses[:10])

min_idx = np.argmin(losses)
optimal_point = grid[min_idx]
print(f"\nQ6 - Optimal point from grid: {optimal_point}")
print(f"Expected: array([42.5, 99.])")

# Gradient Descent
learning_rate = 0.1
n_iterations = 100
a = 0
b = 0
m = len(y)

for i in range(n_iterations):
    y_pred = a * X.flatten() + b
    d_a = (2/m) * np.sum((y_pred - y) * X.flatten())
    d_b = (2/m) * np.sum(y_pred - y)
    a = a - learning_rate * d_a
    b = b - learning_rate * d_b

print(f"\nQ7 - Gradient Descent Results:")
print(f"Coefficients (a): {a}")
print(f"Intercept (b): {b}")
print(f"Expected a: 42.61943031121358")
print(f"Expected b: 99.18581814447936")

# Scikit-learn comparison
reg = LinearRegression()
reg.fit(X, y)
print(f"\nQ9 - Scikit-learn Results:")
print(f"Coefficients: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")
print(f"Expected Coefficients: [42.61943029]")
print(f"Expected Intercept: 99.18581817296929")

print("\n" + "=" * 80)
print("AUDIT VERIFICATION COMPLETE")
print("=" * 80)
