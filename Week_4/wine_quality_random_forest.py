# ============================================================
# Project: Wine Quality Classification using Random Forest
# Description: Predict wine quality (multiclass classification)
# ============================================================

# ==============================
# Import required libraries
# ==============================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Load dataset
# ==============================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# ==============================
# Split features and target
# ==============================
X = data.drop("quality", axis=1)
y = data["quality"]

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Model
# ==============================
model = RandomForestClassifier(n_estimators=200, max_depth=7, class_weight="balanced", random_state=42)

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
print(f"Accuracy; {accuracy*100:.2f}%")

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