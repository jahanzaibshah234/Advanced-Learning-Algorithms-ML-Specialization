# Project Title: "Predict Exam Scores from Study Hours"

"""Problem Statement:
You're given study hours for students and their corresponding exam scores. 
Build a neural network model using tf.keras.Sequential 
and Dense layers to learn the pattern and predict future exam scores."""

# Hides all TF logs except errors
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3' 

# Import requried libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Hours studied vs exam scores
X_train = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=float)
y_train = np.array([35, 50, 55, 60, 65, 70, 75, 85, 90], dtype=float)

# Normalize the input features (study hours)
x_mean = X_train.mean()
x_std = X_train.std()
x_train_norm = (X_train - x_mean) / x_std

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1) # No activation = linear
])

# Configure the model for training
model.compile(optimizer='sgd', loss="mean_squared_error")

# Train the model
model.fit(x_train_norm, y_train, epochs=500, verbose=0)

# Predict Future Exam Score
new_hours = np.array([10, 11, 12], dtype=float)
new_hours_norm = (new_hours - x_mean) / x_std
predictions = model.predict(new_hours_norm)

# Print the predictions
for h, s in zip(new_hours, predictions):
    print(f"Prediction for Future exam score: {h}  → Exam Score: {s[0]:.2f}")

# Plot training data
plt.scatter(X_train, y_train, color="red", label="Actual Data")

# Plot model predictions
x_new = np.linspace(1, 12, 100)
x_new_norm = (x_new - x_mean) / x_std
y_pred = model.predict(x_new_norm)
plt.plot(x_new, y_pred, color="blue", label="Model Predictions")

# Plot new predictions
plt.scatter(new_hours, predictions, color="green", label="New Predictions")

# Add plot labels and formatting
plt.xlabel("Study Hours")
plt.ylabel("Exam Scores")
plt.title("Predict Exam Scores from Study Hours")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()