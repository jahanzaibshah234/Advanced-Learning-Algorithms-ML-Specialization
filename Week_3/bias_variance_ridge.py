import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- DATA GENERATION ---
# Generating synthetic data with some noise
np.random.seed(1)
x = np.random.rand(500, 1) * 20
y = (x -10)**2 * 3 + x * 5 + np.random.rand(500, 1) * 40

# Split the data (60% Train, 20% CV, 20% Test)
# First split: Separate out the 20% Test set
x_temp, x_test, y_temp, y_test = train_test_split(x, y, test_size=0.20, random_state=1)

# Second split: Split the remaining 80% into Train (60% total) and CV (20% total)
x_train, x_cv, y_train, y_cv = train_test_split(x_temp, y_temp, test_size=0.25, random_state=1)

print(f"Train: {x_train.shape}, CV: {x_cv.shape}, Test: {x_test.shape}")

# Setup High Variance Data (Degree 10)
poly = PolynomialFeatures(degree=10, include_bias=False)
X_train_poly = poly.fit_transform(x_train)
X_cv_poly = poly.transform(x_cv)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_cv_scaled = scaler.transform(X_cv_poly)

# --- RIDGE REGULARIZATION ANALYSIS ---
# List of alphas
alphas = [0.001, 0.01, 0.1, 1, 10, 100]
train_errors = []
cv_errors = []

print("Testing Ridge regularization with polynomial features...")

for alpha in alphas:
  # Initialize Ridge with specific alpha
  reg_model = Ridge(alpha=alpha)

  # Fit the model
  reg_model.fit(X_train_scaled, y_train)

  # Predict and Record Error
  yhat_train = reg_model.predict(X_train_scaled)
  yhat_cv = reg_model.predict(X_cv_scaled)

  train_errors.append(mean_squared_error(y_train, yhat_train))
  cv_errors.append(mean_squared_error(y_cv, yhat_cv))

# Find optimal alpha
optimal_alpha_index = np.argmin(cv_errors)
optimal_alpha = alphas[optimal_alpha_index]

# Plotting
plt.plot(np.log10(alphas), train_errors, label='Train MSE', marker='o')
plt.plot(np.log10(alphas), cv_errors, label='CV MSE', marker='o')
plt.axvline(np.log10(optimal_alpha), color='red', linestyle='--', 
            label=f'Optimal alpha: {optimal_alpha}')
plt.xlabel('Log10(Alpha)')
plt.ylabel('Error')
plt.title('Selecting Lambda: Balancing Bias and Variance')
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal alpha: {optimal_alpha}")

print("\n--- Test Set Evaluation ---")
# Combine training and cross-validation data
x_train_full = np.vstack((x_train, x_cv))
y_train_full = np.vstack((y_train, y_cv))

# Feature Engineering with the best degree
poly_final = PolynomialFeatures(degree=10, include_bias=False)
X_train_full_poly = poly_final.fit_transform(x_train_full)
X_test_poly = poly_final.transform(x_test)

# Scaling the features
scaler_final = StandardScaler()
X_train_full_scaled = scaler_final.fit_transform(X_train_full_poly)
X_test_scaled = scaler_final.transform(X_test_poly)

# Train the final model on the full training data
model_final = Ridge(alpha=optimal_alpha)
model_final.fit(X_train_full_scaled, y_train_full)

# Evaluate on the Test Set
yhat_test = model_final.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, yhat_test)

print(f"Final Test MSE with Alpha {optimal_alpha}: {test_mse}")