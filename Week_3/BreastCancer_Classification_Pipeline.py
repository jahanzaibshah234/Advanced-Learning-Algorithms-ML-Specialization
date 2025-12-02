# -------------------------------------------------------
# PROJECT: Breast Cancer Classification with Model Selection,
#          Bias/Variance Diagnosis, Regularization & Learning Curves
# -------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

print("Dataset Loaded:", X.shape, "Labels:", np.unique(y))

# -------------------------------------------------------
# Train / CV / Test Split
# -------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape}, CV: {X_cv.shape}, Test: {X_test.shape}")

# -------------------------------------------------------
# Feature Scaling
# -------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_cv_scaled = scaler.transform(X_cv)
X_test_scaled = scaler.transform(X_test)

# -------------------------------------------------------
# Model Training (Try Different Regularization Strengths)
# -------------------------------------------------------
C_values = [0.01, 0.1, 1, 10]
cv_accuracies = []

for C in C_values:
  model = LogisticRegression(C=C, max_iter=500)
  model.fit(X_train_scaled, y_train)

  preds_cv = model.predict(X_cv_scaled)
  acc = accuracy_score(y_cv, preds_cv)

  cv_accuracies.append(acc)
  print(f"C={C}: CV Accuracy = {acc:.4f}")

# Picking best C
best_index = np.argmax(cv_accuracies)
best_C = C_values[best_index]
print("\nBest C based on CV accuracy:", best_C)

# -------------------------------------------------------
# Retrain Best Model on Training Set
# -------------------------------------------------------
best_model = LogisticRegression(C=best_C, max_iter=500)
best_model.fit(X_train_scaled, y_train)

train_acc = accuracy_score(y_train, best_model.predict(X_train_scaled))
cv_acc = accuracy_score(y_cv, best_model.predict(X_cv_scaled))

print("Train Accuracy:", train_acc)
print("CV Accuracy:", cv_acc)

# -------------------------------------------------------
# Bias / Variance Diagnosis
# -------------------------------------------------------
print("\n--- Bias / Variance Diagnosis ---")

if train_acc < 0.96:
  print(f"Both accuracies low: Train {train_acc:.1%}, CV {cv_acc:.1%}")
  print("High Bias (Underfitting)")
elif (train_acc - cv_acc) > 0.05:
  print("High Variance (Overfitting)")
  print(f"Train {train_acc:.1%} >> CV {cv_acc:.1%}")
else:
  print(f"Train {train_acc:.1%}, CV {cv_acc:.1%}")
  print("Good balance - no major bias/variance problem.")

# -------------------------------------------------------
# Learning Curve
# -------------------------------------------------------
train_sizes = [20, 50, 100, 200, len(X_train)]

train_scores = []
cv_scores = []

for size in train_sizes:
  X_part = X_train_scaled[:size]
  y_part = y_train[:size]

  model = LogisticRegression(C=best_C, max_iter=500)

  model.fit(X_part, y_part)

  train_scores.append(accuracy_score(y_part, model.predict(X_part)))
  cv_scores.append(accuracy_score(y_cv, model.predict(X_cv_scaled)))

# Plot Learning Curves
plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores, marker="o", label="Training Accuracy")
plt.plot(train_sizes, cv_scores, marker="o", label="CV Accuracy")

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.title("Learning Curve")
plt.legend()
plt.grid(True)
plt.show()

# -------------------------------------------------------
# Error Analysis (CV set)
# -------------------------------------------------------
pred_cv = best_model.predict(X_cv_scaled)
wrong_indices = np.where(pred_cv != y_cv)[0]

print("\nMisclassified Samples (first 10):")
print(pd.DataFrame(X_cv[wrong_indices[:10]], columns=feature_names).head(10))
print(f"Error type: {'benign→malignant' if y_cv[wrong_indices[0]]==1 else 'malignant→benign'}")

print("\nConfusion Matrix on CV set:")
print(confusion_matrix(y_cv, pred_cv))

# -------------------------------------------------------
# Decision: What to Try Next
# -------------------------------------------------------
print("\n--- What to Try Next ---")

model_is_good = False

if train_acc < 0.96:
  print("High Bias (Underfitting)")
  print("Suggestions: Complex model, less regularization, more features")
elif (train_acc - cv_acc) > 0.05:
  print("HIGH VARIANCE (Overfitting)")
  print("Suggestions: More regularization, more data, simpler model")
else:
  print("Model is performing well")
  print("Suggestions: Hyperparameter tuning, try other algorithms")
  model_is_good = True

# -------------------------------------------------------
# Final Test Set Evaluation
# -------------------------------------------------------
print("\n--- Final Test Set Evaluation ---")

if model_is_good:
  preds_test = best_model.predict(X_test_scaled)
  test_acc = accuracy_score(y_test, preds_test)

  print(f"CV Accuracy: {cv_acc*100:.1f}%")
  print(f"Test Accuracy: {test_acc*100:.1f}%")

  if (cv_acc - test_acc) > 0.05:
    print("\nLarge drop in Test Accuracy! (Validation Overfitting)")
  else:
    print("Test score is consistent with CV.")
else:
  print("SKIPPING TEST SET: Fix Bias/Variance issues first.")


print("\n--- END OF PROJECT ---")