# ============================================================
# Project: House Price Prediction using Random Forest Regressor
# Description: Predict house prices using Random Forest Regression
# ============================================================

# ==============================
# Import required libraries
# ==============================
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# Load dataset
# ==============================
data = fetch_california_housing()
X = data.data
y = data.target

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Initialize Random Forest Regressor
# ==============================
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)

# ==============================
# Train model
# ==============================
model.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================
preds = model.predict(X_test)

# ==============================
# Evaluation
# ==============================
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# ==============================
# Feature Importance
# ==============================
importances = pd.Series(
    model.feature_importances_, index=data.feature_names
).sort_values(ascending=False)

print("\n=== Feature Importance ===")
print(importances.head(10))
