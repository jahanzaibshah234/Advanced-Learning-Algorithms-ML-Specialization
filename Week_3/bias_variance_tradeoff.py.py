import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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

# Iterate through Polynomial Degrees
train_mses = []
cv_mses = []
degree = range(1, 11)

for d in degree:
  # Feature Engineering
  poly = PolynomialFeatures(degree=d, include_bias=False)

  # Transform Train and CV
  X_train_poly = poly.fit_transform(x_train)
  X_cv_poly = poly.transform(x_cv)

  # Scaling
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train_poly)
  X_cv_scaled = scaler.transform(X_cv_poly)

  # Training
  model = LinearRegression()
  model.fit(X_train_scaled, y_train)

  # Evaluation
  yhat_train = model.predict(X_train_scaled)
  yhat_cv = model.predict(X_cv_scaled)

  # Calculate MSE
  train_mses.append(mean_squared_error(y_train, yhat_train))
  cv_mses.append(mean_squared_error(y_cv, yhat_cv))

# Find the degree with the minimum CV MSE
min_cv_mse = min(cv_mses)
best_degree_index = cv_mses.index(min_cv_mse)
best_degree = degree[best_degree_index]

# Visualization
plt.plot(degree, train_mses, label='Train MSE (J_Train)', marker='o')
plt.plot(degree, cv_mses, label='CV MSE (J_CV)', marker='o')
plt.axvline(best_degree, color='red', linestyle='--', 
            label=f'Optimal Degree: {best_degree}')
plt.xlabel('Degree of Polynomial')
plt.ylabel('Mean Squared Error')
plt.title('Model Selection: The U-Curve')
plt.legend()
plt.grid(True)
plt.show()

print(f"\nOptimal polynomial degree based on CV error: {best_degree}")

print("--- Test Set Evaluation ---")
# Combine training and cross-validation data
x_train_full = np.vstack((x_train, x_cv))
y_train_full = np.vstack((y_train, y_cv))

# Feature Engineering with the best degree
poly_final = PolynomialFeatures(degree=best_degree, include_bias=False)
X_train_full_poly = poly_final.fit_transform(x_train_full)
X_test_poly = poly_final.transform(x_test)

# Scaling the features
scaler_final = StandardScaler()
X_train_full_scaled = scaler_final.fit_transform(X_train_full_poly)
X_test_scaled = scaler_final.transform(X_test_poly)

# Train the final model on the full training data
model_final = LinearRegression()
model_final.fit(X_train_full_scaled, y_train_full)

# Evaluate on the Test Set
yhat_test = model_final.predict(X_test_scaled)
test_mse = mean_squared_error(y_test, yhat_test)

print(f"Mean Squared Error on the Test Set: {test_mse}")