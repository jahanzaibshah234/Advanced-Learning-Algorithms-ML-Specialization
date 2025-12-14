# ===== Project: Error Metrics for Skewed Binary Classification =====

import os
# ==================================
# Hides all TF logs except errors
# ==================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.random.set_seed(42)

# ==================================
# Generate imbalanced dataset
# ==================================

n = 3000
X = np.random.randn(n, 1)
y = (X[:, 0] > 2).astype(int)   # very few positives

# ==================================
# Split data (80/20)
# ==================================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# ==================================
# Standardize features
# ==================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ==================================
# Build simple model
# ==================================

model = tf.keras.Sequential([
  tf.keras.Input(shape=(1,)),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# ==================================
# Train model
# ==================================
model.fit(X_train_scaled, y_train, epochs=30, verbose=0)

# ==================================
# Predict probabilities on validation set
# ==================================
y_pred = model.predict(X_val_scaled, verbose=0)

# ==================================
# Evaluate model at different thresholds (precision–recall tradeoff)
# ==================================
thresholds = [0.2, 0.3, 0.5, 0.7]

for t in thresholds:
  y_pred_binary = (y_pred > t).astype(int)
  print(f"\nThreshold {t}")

  cm = confusion_matrix(y_val, y_pred_binary)
  precision = precision_score(y_val, y_pred_binary, zero_division=0)
  recall = recall_score(y_val, y_pred_binary)
  f1 = f1_score(y_val, y_pred_binary)

# ==================================
# Output 
# ==================================
  print("Confusion Matrix:\n", cm)
  print("Precision:", precision)
  print("Recall:", recall)
  print("F1 Score:", f1)
