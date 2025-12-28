# ============================================================
# Project: Bank Customer Churn Prediction using Random Forest
# Description: Predict whether a customer will exit the bank
# ============================================================

# ==============================
# Import required libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Load dataset
# ==============================
url = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv"
data = pd.read_csv(url)

# ==============================
# Select features
# ==============================
features = ["CreditScore", "Geography", "Gender", "Age", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"]
X = data[features].copy()
y = data["Exited"]

# ==============================
# One-Hot Encoding
# ==============================
X = pd.get_dummies(X)

# ==============================
# Split into train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Model
# ==============================
model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced')
model.fit(X_train, y_train)

# ==============================
# Predict
# ==============================
preds = model.predict(X_test)

# ==============================
# Evaluation
# ==============================
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy*100:.2f}%")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, preds))

print("\n=== Classification Report ===")
print(classification_report(y_test, preds))

# ==============================
# Feature Importance
# ==============================
importances = pd.Series(
    model.feature_importances_, index=X.columns
).sort_values(ascending=False)

print("\n=== Feature Importance ===")
print(importances.head(10))
