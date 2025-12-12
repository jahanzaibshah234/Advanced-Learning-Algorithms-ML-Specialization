# ===== PROJECT 1B: Adding Data (Noise Augmentation TensorFlow Version) =====

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
random.seed(42)
tf.random.set_seed(42)
np.random.seed(42)

# ============================
# 1. Load & Split
# ============================
data = load_breast_cancer()
X = data.data
y = data.target

# Train/validation split (80/20)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)

# ============================
# 2. Feature Scaling
# ============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# ============================
# 3. Data Augmentation
# ============================
noise = np.random.normal(0, 0.05, size=X_train_scaled.shape)
X_aug = X_train_scaled + noise
y_aug = y_train

# Combine original + augmented data
X_combined = np.vstack([X_train_scaled, X_aug])
y_combined = np.hstack([y_train, y_aug])

# ============================
# 4. Build Model
# ============================
model = tf.keras.Sequential([
    tf.keras.Input(shape=(30,)),
     tf.keras.layers.Dense(units=16, activation='relu'),
    tf.keras.layers.Dense(units=8, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# ============================
# 5. Compile & Train
# ============================
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_combined, y_combined, epochs=40, verbose=0)

# ============================
# 6. Evaluate
# ============================
loss, accuracy = model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"Validation Accuracy: {accuracy:.4f}")
