# ============================================================
# Project: Bank Customer Churn Prediction using XGBoost
# Description: Predict whether a customer will exit the bank
# ============================================================

# ==============================
# Import required libraries
# ==============================
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Load churn dataset
# ==============================
url = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv"
data = pd.read_csv(url)


# ==============================
# Select features and target
# ==============================
features = ["CreditScore", "Geography", "Gender", "Age", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"]
X = data[features].copy()
y = data["Exited"]

# ==============================
# Encode categorical variables
# ==============================
X = pd.get_dummies(X)

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# Initialize XGBoost Classifier
# ==============================
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train.value_counts()[0] / y_train.value_counts()[1]),
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
# Evaluation
# ==============================
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy*100:.2f}%")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, preds))

print("\n=== Classification Report ===")
print(classification_report(y_test, preds))