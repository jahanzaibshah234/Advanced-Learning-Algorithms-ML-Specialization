# ===== Project: Fairness, Bias & Ethics in ML =====

import os
# ============================
# Hides all TF logs except errors
# ============================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

tf.random.set_seed(42)

# =====================================================
# 1. CREATE BIASED DATASET
# =====================================================

np.random.seed(42)
n = 1200

# Feature
X = np.random.randn(n, 1)

# Sensitive attribute (0 = Group A, 1 = Group B)
group = np.random.binomial(1, 0.5, n)

# Biased label generation
y = ((X[:, 0] + group * 0.8) > 0).astype(int)

# =====================================================
# Split data (80/20)
# =====================================================
X_train, X_val, y_train, y_val, g_train, g_val = train_test_split(X, y, group, test_size=0.20, random_state=42)


# =====================================================
# Standardize features
# =====================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# =====================================================
# Build model
# =====================================================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=6, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# =====================================================
# Train model
# =====================================================
history = model.fit(X_train_scaled, y_train, epochs=40, validation_data=(X_val_scaled, y_val), verbose=0)

# =====================================================
# Predict on validation data
# =====================================================
y_pred = (model.predict(X_val_scaled, verbose=0) > 0.5).astype(int).flatten()

# =====================================================
# Calculate accuracy per group
# =====================================================
acc_group_0 = np.mean(y_pred[g_val == 0] == y_val[g_val == 0])
acc_group_1 = np.mean(y_pred[g_val == 1] == y_val[g_val == 1])

# =====================================================
# Fairness diagnosis
# =====================================================
gap = abs(acc_group_0 - acc_group_1)

if gap > 0.10:
  fairness = "Biased Model"
else:
  fairness = "Fair Model"

if fairness == "Biased Model":
  suggestion = "Collect balanced data or remove sensitive bias"
else:
  suggestion = "Model is reasonably fair"

# =====================================================
# Output
# =====================================================
print("Accuracy Group 0:", round(acc_group_0, 3))
print("Accuracy Group 1:", round(acc_group_1, 3))
print("Fairness Diagnosis:", fairness)
print("Ethical Recommendation:", suggestion)


