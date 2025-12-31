# ============================================================
# Project: Heart Disease Prediction using XGBoost
# Description: Predict presence of heart disease
# ============================================================

# ==============================
# Import libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance

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
# Initialize XGBoost Classifier
# ==============================
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4,
    random_state=42,
    eval_metric="logloss"
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
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy*100:.2f}%")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, preds))

print("\n=== Classification Report ===")
print(classification_report(y_test, preds))

# ==============================
# Feature Importance
# ==============================
plot_importance(model, max_num_features=10)
plt.show()