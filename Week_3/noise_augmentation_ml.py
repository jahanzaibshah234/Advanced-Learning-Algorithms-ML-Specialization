# ===== PROJECT: Adding Data (Noise Augmentation for ML) =====

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ============================
# 1. Load & Split Dataset
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
# 4. Train Model
# ============================
model = LogisticRegression(max_iter=5000)
model.fit(X_combined, y_combined)

# ============================
# 5. Evaluate
# ============================
y_pred = model.predict(X_val_scaled)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred):.4f}")
