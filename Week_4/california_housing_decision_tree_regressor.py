# ============================================================
# Project: California Housing Price Prediction using Decision Tree Regression
# Description: Predict median house values using Decision Tree Regressor
# ============================================================

# ==============================
# Import Libraries
# ==============================
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ==============================
# Load data
# ==============================
data = fetch_california_housing()
X = data.data
y = data.target

# ==============================
# Split into train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ==============================
# Model
# ==============================
model = DecisionTreeRegressor(max_depth=5)
model.fit(X_train, y_train)

# ==============================
# Predict
# ==============================
preds = model.predict(X_test)

# ==============================
# Evaluation
# ==============================
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, preds)

# ==============================
# Output
# ==============================
print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R² Score: {r2:.3f}")

# ==============================
# Visualize Decision Tree
# ==============================
plt.figure(figsize=(20, 10))
plot_tree(
    model, 
    filled=True, 
    feature_names=data.feature_names, 
    fontsize=6
)

plt.show()