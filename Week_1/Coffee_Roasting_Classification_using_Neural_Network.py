# Project Title: "Coffee Roasting Classification using Neural Network"
"""Problem Statement:
You are given data of coffee roasting temperature (°C) and time (minutes).
You need to build a binary classifier to predict whether the roast is "Good" (1) or 
"Bad" (0) using a Neural Network with TensorFlow."""

# Hides all TF logs except errors
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import requried libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Temperature (°C) and Time (minutes) vs Roast Level
X_train = np.array([
    [180, 11],
    [200, 12],
    [220, 14],
    [160, 10],
    [210, 13],
    [170, 11]
], dtype=float)
Y_train = np.array([0, 1, 1, 0, 1, 0], dtype=float)


# Normalize the input features (Temperature (°C) and Time (minutes))
X_mean = np.mean(X_train, axis=0)
X_std = np.std(X_train, axis=0)
X_train_norm = (X_train - X_mean) / X_std

# Build the neural network model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(2, )),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Configure the model for training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_norm, Y_train, epochs=500, verbose=0)

# Predict New Sample
new_sample = np.array([[205.0, 12.5]])
new_sample_norm = (new_sample - X_mean) / X_std
predictions = model.predict(new_sample_norm)

# Print the predictions
for t, p in zip(new_sample, predictions):
    result = "Good Roast (1)" if p[0] >= 0.5 else "Bad Roast (0)"
    print(f"Sample {t} → Probability: {p[0]:.2f} → {result}")

# Plot training data
plt.scatter(X_train[:, 0], Y_train, color="red", label="Actual Data")

# Plot Prediction Curve (fixing Time at 12.0 mins)
X_temp = np.linspace(160, 220, 100)
X_temp_norm = (X_temp - X_mean[0]) / X_std[0]
X_time_norm = (12.0 - X_mean[1]) / X_std[1]

# Create Input Matrix with Temp varying and Time fixed
X_time_norm_repeated = np.full_like(X_temp_norm, X_time_norm)
X_predict_norm = np.column_stack((X_temp_norm, X_time_norm_repeated))

# Model Predictions
Y_pred = model.predict(X_predict_norm)
plt.plot(X_temp, Y_pred, color="blue", label="Model Predictions")

# Plot new predictions
plt.scatter(new_sample[:, 0], predictions, color="green", label="New Predictions")

# Add plot labels and formatting
plt.xlabel("Temperature (°C)")
plt.ylabel("Roast Quality (Good=1 / Bad=0)")
plt.title("Coffee Roasting Classification")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot Accuracy vs Epochs
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()