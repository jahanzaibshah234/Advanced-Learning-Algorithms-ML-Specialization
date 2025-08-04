# Hides all TF logs except errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Import requried libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Coffee temprature vs roast level
X_train = np.array([170, 180, 190, 200, 210, 220, 230, 240], dtype=float)
Y_train = np.array([0.1, 0.15, 0.25, 0.4, 0.6, 0.75, 0.9, 0.95], dtype=float)

# Normalize the input features (temperature)
X_mean = X_train.mean()
X_std = X_train.std()
X_train_normalized = (X_train - X_mean) / X_std

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

# Configure the model for training
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X_train_normalized, Y_train, epochs=500, verbose=0)

# Predict new roast level
new_roast = np.array([185, 205, 225], dtype=float)
new_roast_normalized = (new_roast - X_mean) / X_std
predictions = model.predict(new_roast_normalized)

# Print the predictions
for t, p in zip(new_roast, predictions):
    print(f"Temp: {t}° → Roast Level: {p[0]:.2f}")

# Plot training data
plt.scatter(X_train, Y_train, color='red', label="Actual Data")

# Plot model predictions
X_new = np.linspace(170, 240, 100)
X_new_normalized = (X_new - X_mean) / X_std
Y_pred = model.predict(X_new_normalized)
plt.plot(X_new, Y_pred, color="blue", label="Model Predictions")

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