# ============================================================
# Project: House Price Prediction using XGBoost Regressor
# Description: Predict house prices using gradient boosting
# ============================================================

# ==============================
# Import libraries
# ==============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor, plot_importance

# ==============================
# Load housing dataset
# ==============================
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ==============================
# Initialize XGBoost Regressor
# ==============================
model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
    eval_metric="rmse"
)

# ==============================
# Train model
# ==============================
model.fit(X_train, y_train)

# ==============================
# Prediction
# ==============================
preds = model.predict(X_test)

# ==============================
# Evaluate model
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
plot_importance(model, max_num_features=10)
plt.show()