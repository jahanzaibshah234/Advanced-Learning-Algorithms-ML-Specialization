# ============================================================
# Project: Wine Quality Classification using XGBoost
# Description: Predict wine quality (multiclass classification)
# ============================================================

# ==============================
# Import libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from xgboost import XGBClassifier, plot_importance

# ==============================
# Load wine dataset
# ==============================
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# ==============================
# Feature-target separation
# ==============================
X = data.drop("quality", axis=1)

# ==============================
# Encode the Target Labels
# ==============================
le = LabelEncoder()
y = le.fit_transform(data["quality"])

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ==============================
# Initialize XGBoost Multiclass Model
# ==============================
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softmax",
    num_class=len(le.classes_),
    random_state=42,
    eval_metric="mlogloss"
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