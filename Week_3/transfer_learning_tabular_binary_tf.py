# ===== Project: Transfer Learning for Tabular Binary Classification Using TensorFlow =====

import os
# ============================
# Hides all TF logs and Lock OS Hash Seed
# ============================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = '0'

# ============================
# Import required libraries
# ============================
import numpy as np
import tensorflow as tf
import random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================
# LOCK ALL SEEDS
# ============================
tf.random.set_seed(42)
random.seed(42)
np.random.seed(42)

# ============================================
# Load Dataset
# ============================================
data = load_breast_cancer()
X = data.data
y = data.target

# ============================================
# Create Pretraining Dataset (Large Data)
# ============================================
X_pre, X_unused, y_pre, y_unused = train_test_split(X, y, test_size=0.20, random_state=42)

# Standardize pretraining data only
scaler_pre = StandardScaler()
X_pre_scaled = scaler_pre.fit_transform(X_pre)

# ============================================
# Create Target (Fine-tuning) Dataset (Low Data)
# ============================================

# Split unused data into - Train/Validation Set
X_train, X_val, y_train, y_val = train_test_split(X_unused, y_unused, test_size=0.50, random_state=42)

# Randomly select a small subset (20%)
indices = np.random.permutation(len(X_train))
k = int(len(X_train) * 0.20)
X_train_small = X_train[indices[:k]]
y_train_small = y_train[indices[:k]]

# Standardize fine-tuning data using a new scaler
scaler_finetune = StandardScaler()
X_train_scaled = scaler_finetune.fit_transform(X_train_small)
X_val_scaled = scaler_finetune.transform(X_val)

print("Small training samples:", len(X_train_small))

# ============================================
# Build Base Neural Network Model
# ============================================
def build_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(30,)),
      tf.keras.layers.Dense(units=16, activation='relu'),
      tf.keras.layers.Dense(units=8, activation='relu'),
      tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model
# ============================================
# Pretrain the Model
# ============================================
model_pre = build_model()
history_pre = model_pre.fit(X_pre_scaled, y_pre, epochs=40, batch_size=32, verbose=0)

# Save learned weights for transfer learning
pretrained_weights = model_pre.get_weights()

# ============================================
# Transfer Learning Setup
# ============================================
# Initialize a new model with the same architecture
# Load pretrained weights
model_transfer = build_model()
model_transfer.set_weights(pretrained_weights)

# ============================================
# Fine-tune on Small Dataset
# ============================================
history_transfer = model_transfer.fit(X_train_scaled, y_train_small, validation_data=(X_val_scaled, y_val), epochs=40, batch_size=16, verbose=0)

# ============================================
# Evaluate Model
# ============================================
loss, acc = model_transfer.evaluate(X_val_scaled, y_val, verbose=0)
print(f"Validation Accuracy (Transfer Learning): {acc:.4f}, {loss:.4f}")

