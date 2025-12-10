# ===== Project: Fixing High Variance Using Regularization & Dropout (TensorFlow) =====

# ============================
# Hides all TF logs except errors
# ============================
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================
# Import required libraries
# ============================
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================
# LOCK ALL SEEDS
# ============================
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ============================
# Generate dataset (make_circles)
# ============================

X, y = make_circles(n_samples=200, noise=0.3, factor=0.3, random_state=42)

# ============================
# Split dataset (80/20)
# ============================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# ============================
# Feature Scaling
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ============================
# Model A — Overfitting model (no regularization)
# ============================
model_overfit = Sequential([
    tf.keras.Input(shape=(2, )),
    Dense(128, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model_overfit.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history_overfit = model_overfit.fit(X_train_scaled, y_train, epochs=200, validation_data=(X_val_scaled, y_val), verbose=0)

# ============================
# Model B — Regularized model (L2 + Dropout)
# ============================
model_regularized = Sequential([
  tf.keras.Input(shape=(2, )),
    Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(units=128, activation='relu', kernel_regularizer=l2(0.01)),
    Dropout(0.3),
    Dense(units=64, activation='relu', kernel_regularizer=l2(0.01)),
    Dense(units=1, activation='sigmoid')
])

# Compile model
model_regularized.compile(optimizer=Adam(0.01), loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history_regularized = model_regularized.fit(X_train_scaled, y_train, epochs=200, validation_data=(X_val_scaled, y_val), verbose=0)

# ============================
# Plot learning curves
# ============================
plt.figure(figsize=(12, 5))

# Overfitting model
plt.subplot(1, 2, 1)
plt.plot(history_overfit.history['accuracy'], label='Train Acc')
plt.plot(history_overfit.history['val_accuracy'], label='Val Acc')
plt.title("Overfitting Model (No Regularization)")
plt.ylim(0.5, 1.05)
plt.legend()
plt.grid(True)

# Regularized model
plt.subplot(1, 2, 2)
plt.plot(history_regularized.history['accuracy'], label='Train Acc')
plt.plot(history_regularized.history['val_accuracy'], label='Val Acc')
plt.title("Regularized Model (L2 + Dropout)")
plt.ylim(0.5, 1.05)
plt.legend()
plt.grid(True)

plt.show()

# ============================
# Bias/Variance Diagnosis
# ============================

final_train_over = history_overfit.history['accuracy'][-1]
final_val_over = history_overfit.history['val_accuracy'][-1]

final_train_reg = history_regularized.history['accuracy'][-1]
final_val_reg = history_regularized.history['val_accuracy'][-1]

# ---- Overfitting Model Diagnosis ----
if final_train_over >= 0.90 and (final_train_over - final_val_over) > 0.10:
  diagnosis_overfit = "High Variance"
elif final_train_over < 0.85 and final_val_over < 0.85:
  diagnosis_overfit = "High Bias"
else:
  diagnosis_overfit = "Good Fit"

# ---- Regularized Model Diagnosis ----
if final_train_reg >= 0.85 and (final_train_reg - final_val_reg) < 0.15:
  diagnosis_regularized = "Good Fit"
elif final_train_reg < 0.85 and final_val_reg < 0.85:
  diagnosis_regularized = "High Bias"
else:
  diagnosis_regularized = "High Variance"


# ============================
# Suggest what to try next
# ============================

if diagnosis_overfit == "High Variance":
  suggestion_overfit = "Increase regularization or reduce model size"
elif diagnosis_overfit == "High Bias":
    suggestion_overfit = "Increase model complexity (more units or layers)"
else:
  suggestion_overfit = "Model is fine"

if diagnosis_regularized == "High Bias":
  suggestion_regularized = "Use a larger model or reduce regularization"
elif diagnosis_regularized == "High Variance":
  suggestion_regularized = "Increase regularization or reduce model size"
else:
  suggestion_regularized = "Model is fine"

# ============================
# Output results
# ============================

print(f"Overfit Model (Train: {final_train_over:.2f}, Val: {final_val_over:.2f}): {diagnosis_overfit}")
print(f"Suggestion: {suggestion_overfit}\n")

print(f"Regularized Model (Train: {final_train_reg:.2f}, Val: {final_val_reg:.2f}): {diagnosis_regularized}")
print(f"Suggestion: {suggestion_regularized}")
