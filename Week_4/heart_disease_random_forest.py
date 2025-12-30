# ============================================================
# Project: Heart Disease Prediction using Random Forest
# Description: Predict presence of heart disease
# ============================================================

# ==============================
# Import required libraries
# ==============================
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Load dataset
# ==============================
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
data = pd.read_csv(url)


# ==============================
# Separate features and target
# ==============================
X = data.drop("target", axis=1)
y = data["target"]

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Model
# ==============================
model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight='balanced', random_state=42)

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