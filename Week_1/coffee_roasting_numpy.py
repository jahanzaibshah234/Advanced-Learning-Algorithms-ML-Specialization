# Import requried libraries
import numpy as np
import matplotlib.pyplot as plt

# Coffee temprature vs roast level
X_train = np.array([170, 180, 190, 200, 210, 220, 230, 240], dtype=float)
Y_train = np.array([0.1, 0.15, 0.25, 0.4, 0.6, 0.75, 0.9, 0.95], dtype=float)

# Normalize the input features (temperature)
X_mean = X_train.mean()
X_std = X_train.std()
X_train_norm = (X_train - X_mean) / X_std

# Initialize weights and bias
w = 0.0
b = 0.0
learning_rate = 0.01
epochs = 500

# Training using gradient descent
for epoch in range(epochs):
    y_pred = w * X_train_norm + b
    error = y_pred - Y_train
    dw = np.mean(error * X_train_norm)
    db = np.mean(error)

    w = w - learning_rate * dw
    b = b - learning_rate * db

# Predict for new values
new_roast = np.array([185, 205, 225], dtype=float)
new_roast_norm = (new_roast - X_mean) / X_std
predictions = w * new_roast_norm + b

# Print the predictions
for t, p in zip(new_roast, predictions):
    print(f"Temp: {t}° → Roast Level: {p:.2f}")

# Plot training data
plt.scatter(X_train, Y_train, color="red", label="Actual Data")

# Line of best fit
X_new = np.linspace(170, 240, 100)
X_new_norm = (X_new - X_mean) / X_std
Y_new_pred = w * X_new_norm + b
plt.plot(X_new, Y_new_pred, color="blue", label="Model Predictions")

# Plot new predictions
plt.scatter(new_roast, predictions, color="green", label="New Predictions")

# Add plot labels and formatting
plt.xlabel("Temperature (°C)")
plt.ylabel("Roast Level")
plt.title("Coffee Roast Level vs Temperature")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()