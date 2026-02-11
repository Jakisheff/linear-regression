from sklearn.linear_model import LinearRegression
import numpy as np

# Training data
X = [[1], [2.1], [3]]
y = [[1], [2], [3]]

# Initialize and fit the model
reg = LinearRegression()
reg.fit(X, y)

# Predict for x_pred = [[4]]
x_pred = [[4]]
prediction = reg.predict(x_pred)

# Print results
print(f"Prediction for {x_pred}: {prediction}")
print(f"Coefficients (coef_): {reg.coef_}")
print(f"Intercept (intercept_): {reg.intercept_}")
print(f"Score: {reg.score(X, y)}")
