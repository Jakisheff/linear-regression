import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# 1. Generate Data
X, y, coef = make_regression(n_samples=100,
                             n_features=1,
                             n_informative=1,
                             noise=10,
                             coef=True,
                             random_state=0,
                             bias=100.0)

# Reshape X to be (100, 1) and y to match X for broadcasting if needed
# But y from make_regression is (100,).
# We will use y as (100,) for calculations.

def compute_mse(coefs, X, y):
    """
    coefs is a list that contains a and b: [a,b]
    X is the features set
    y is the target
    Returns a float which is the MSE
    """
    a, b = coefs
    # y_pred = a*x + b
    # X is (100, 1)
    y_preds = a * X.flatten() + b
    mse = np.mean((y_preds - y)**2)
    return mse

# 2. Compute MSE for a=1, b=2
print(f"MSE for a=1, b=2: {compute_mse([1, 2], X, y)}")

# 3. Create a grid of 640000 points
aa, bb = np.mgrid[-200:200:0.5, -200:200:0.5]
grid = np.c_[aa.ravel(), bb.ravel()]

# 4. Compute MSE for all points in the grid
# To speed up, we can use vectorized operations instead of a loop
# y_preds = a*X + b
# We want to compute mean((a*X + b - y)^2) for many a, b.
# Let's define a function that works on the flattened grid.
# However, the instructions verify 640000 points, so let's stick to a loop or efficient map.

losses = []
# Using loop for clarity and to match instructions roughly, but optimization is better.
# given 640,000 points, a simple loop might be slow in Python.
# Let's vectorise:
# loss = mean((a*x + b - y)^2) = mean((a*x + b)^2 - 2*y*(a*x+b) + y^2)
# This is still a bit complex to vectorize fully without broadcasting large arrays.
# Let's trust numpy's speed or use a simple loop.
# Actually, the grid is (800, 800) = 640,000. 
# A loop in Python might take a few seconds.

print("Computing losses for grid...")
# Vectorized approach:
# X_flat = X.flatten() # (100,)
# grid shape: (640000, 2)
# We can do this in batches if needed, or just iterate.
for i in range(len(grid)):
    losses.append(compute_mse(grid[i], X, y))

losses_reshaped = np.array(losses).reshape(aa.shape)

# 5. Plot MSE in 2D
f, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(aa,
                    bb,
                    losses_reshaped,
                    100,
                    cmap="RdBu",
                    vmin=0,
                    vmax=160000)
ax_c = f.colorbar(contour)
ax_c.set_label("MSE")

ax.set(aspect="equal",
    xlim=(-200, 200),
    ylim=(-200, 200),
    xlabel="$a$",
    ylabel="$b$")

# Find optimal a and b from grid
min_idx = np.argmin(losses)
optimal_a_grid = grid[min_idx][0]
optimal_b_grid = grid[min_idx][1]
print(f"Optimal a (grid): {optimal_a_grid}, Optimal b (grid): {optimal_b_grid}")

# Plot the optimal point from grid
ax.scatter(optimal_a_grid, optimal_b_grid, c='green', s=100, label='Grid Minimum')


# 6. Gradient Descent
# y_pred = a*x + b
# Loss = (1/N) * sum((y_pred - y)^2)
# dLoss/da = (2/N) * sum((y_pred - y) * x)
# dLoss/db = (2/N) * sum(y_pred - y)

learning_rate = 0.1 # This might be too high for unscaled data, usually needs tuning. 
# The exercise says learning rate = 0.1. But make_regression data might have large values.
# Let's follow instructions as is, but be aware of divergence.
# If it diverges, usually we lower the learning rate.
# Also, n_features=1, noise=10, bias=100.
# X is random.

n_iterations = 100
a = 0
b = 0
m = len(y)
history = []

print("Starting Gradient Descent...")
for i in range(n_iterations):
    y_pred = a * X.flatten() + b
    
    # Gradients
    d_a = (2/m) * np.sum((y_pred - y) * X.flatten())
    d_b = (2/m) * np.sum(y_pred - y)
    
    # Update
    a = a - learning_rate * d_a
    b = b - learning_rate * d_b
    
    history.append([a, b])

history = np.array(history)
print(f"Gradient Descent Optimal a: {a}, Optimal b: {b}")

# Plot path
ax.plot(history[:, 0], history[:, 1], color='yellow', marker='.', linestyle='-', label='Gradient Descent Path')
ax.scatter(history[-1, 0], history[-1, 1], c='orange', s=100, label='GD Final')

plt.legend()
plt.savefig('ex05_loss_surface_params.png')
print("Loss surface plot saved to ex05_loss_surface_params.png")


# 7. Compare with Scikit-learn
reg = LinearRegression()
reg.fit(X, y)
print(f"Scikit-learn Optimal a: {reg.coef_[0]}, Optimal b: {reg.intercept_}")

# Add Scikit-learn point to plot
# We would need to reload the figure or just add it before saving.
# It's fine to just print it for comparison as requested.
