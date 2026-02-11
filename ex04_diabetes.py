from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

# Load the dataset
diabetes = load_diabetes(as_frame=True)
X, y = diabetes.data, diabetes.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)

# Fit Linear Regression
reg = LinearRegression()
reg.fit(X_train, y_train)

# Coefficients and Intercept
print("Coefficients:", reg.coef_)
print("Intercept:", reg.intercept_)

# Equation (simplified representation)
print("Equation: y = " + " + ".join([f"{c:.2f}*x{i}" for i, c in enumerate(reg.coef_)]) + f" + {reg.intercept_:.2f}")

# Predict on test set
y_pred = reg.predict(X_test)
print("\nPredictions on test set (first 5):", y_pred[:5])

# Compute MSE
mse_train = mean_squared_error(y_train, reg.predict(X_train))
mse_test = mean_squared_error(y_test, y_pred)

print(f"\nMSE on Train Set: {mse_train:.4f}")
print(f"MSE on Test Set: {mse_test:.4f}")
