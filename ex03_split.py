import numpy as np
from sklearn.model_selection import train_test_split

# Create data
X = np.arange(1, 21).reshape(10, -1)
y = np.arange(1, 11)

print("Original X shape:", X.shape)
print("Original y shape:", y.shape)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Print results
print("\nX_train:\n", X_train)
print("\ny_train:\n", y_train)
print("\nX_test:\n", X_test)
print("\ny_test:\n", y_test)
