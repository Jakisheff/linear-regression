import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def compute_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def run_experiment(noise_level):
    print(f"\n--- Running Experiment with Noise Level = {noise_level} ---")
    
    # 1. Generate data
    X, y, coef = make_regression(n_samples=100,
                                 n_features=1,
                                 n_informative=1,
                                 noise=noise_level,
                                 coef=True,
                                 random_state=0,
                                 bias=100.0)
    
    # 2. Plot the data
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data')
    plt.title(f'Linear Regression Data (Noise={noise_level})')
    plt.xlabel('X')
    plt.ylabel('y')
    
    # 3. Fit Linear Regression
    reg = LinearRegression()
    reg.fit(X, y)
    
    print(f"Model Equation: y = {reg.coef_[0]:.4f} * x + {reg.intercept_:.4f}")
    
    # 4. Add fitted line to plot
    x_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_range_pred = reg.predict(x_range)
    
    plt.plot(x_range, y_range_pred, color='red', linewidth=2, label='Fitted Line')
    plt.legend()
    plt.savefig(f'ex02_plot_noise_{noise_level}.png')
    print(f"Plot saved to ex02_plot_noise_{noise_level}.png")
    
    # 5. Predict on X
    y_pred = reg.predict(X)
    
    # 6. Compute MSE
    mse = compute_mse(y, y_pred)
    print(f"Mean Squared Error (MSE): {mse:.4f}")

if __name__ == "__main__":
    # First run with noise=10
    run_experiment(noise_level=10)
    
    # Second run with noise=50
    run_experiment(noise_level=50)
