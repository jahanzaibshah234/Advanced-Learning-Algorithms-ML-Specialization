# ===== Project: Bias vs Variance in Neural Networks (TensorFlow) =====

import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

random.seed(42)
tf.random.set_seed(42)

# ============================
# Generate dataset (make_moons)
# ============================
X, y = make_moons(n_samples=2000, noise=0.2, random_state=42)

# ============================
# Split into train / val (80/20)
# ============================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)   # TODO

# ============================
# Feature Scaling
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ============================
# Build Model A (Underfitting model)
# ============================
model_small = Sequential([
    Dense(units=4, activation='relu', input_dim=2),
    Dense(units=1, activation='sigmoid')
])  # TODO build Sequential model

# Compile model
model_small.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history_small = model_small.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_val_scaled, y_val), verbose=0)

# ============================
# Build Model B (Overfitting model)
# ============================

model_large = Sequential([
    Dense(units=64, activation='relu', input_dim=2),
    Dense(units=64, activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=1, activation='sigmoid')
]) 

# Compile model
model_large.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history_large = model_large.fit(X_train_scaled, y_train, epochs=50, validation_data=(X_val_scaled, y_val), verbose=0)

# ============================
# Plot learning curves for both models
# ============================
plt.figure(figsize=(12, 5))

# Small model
plt.subplot(1, 2, 1)
plt.plot(history_small.history['accuracy'], label='Train Acc')
plt.plot(history_small.history['val_accuracy'], label='Val Acc')
plt.title("Small Model (Underfitting)")
plt.grid(True)
plt.legend()

# Large model
plt.subplot(1, 2, 2)
plt.plot(history_large.history['accuracy'], label='Train Acc')
plt.plot(history_large.history['val_accuracy'], label='Val Acc')
plt.title("Large Model (Overfitting)")
plt.grid(True)
plt.legend()

plt.show()

# ============================
# Bias/Variance Diagnosis
# ============================

final_train_small = history_small.history['accuracy'][-1]
final_val_small = history_small.history['val_accuracy'][-1]

final_train_large = history_large.history['accuracy'][-1]
final_val_large = history_large.history['val_accuracy'][-1]

# Diagnose small model
if final_train_small < 0.80 and final_val_small < 0.80:
  diagnosis_small = "High Bias" 
else:
  diagnosis_small = "Good Fit"

# Diagnose large model
if final_train_large > 0.97 and (final_train_large - final_val_large) > 0.15:
  diagnosis_large = "High Variance"
else:
  diagnosis_large = "Good Fit"

# ============================
# Suggest what to try next
# ============================
if diagnosis_small == "High Bias":
  suggestion_small = "Increase model size or train longer"
else:
  suggestion_small = "Model is fine"

if diagnosis_large == "High Variance":
  suggestion_large = "Add regularization (L2) or reduce model size"
else:
  suggestion_large = "Model is fine"

# ============================
# Output
# ============================

print("Small Model Diagnosis:", diagnosis_small)
print("Large Model Diagnosis:", diagnosis_large)

print("Suggestions:")
print("Small model:", suggestion_small)
print("Large model:", suggestion_large)
