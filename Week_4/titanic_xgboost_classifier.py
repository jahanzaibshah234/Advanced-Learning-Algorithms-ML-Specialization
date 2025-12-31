# ============================================================
# Project: Titanic Survival Prediction using XGBoost
# Description: Predict passenger survival
# ============================================================

# ==============================
# Import required libraries
# ==============================
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Load Titanic dataset
# ==============================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# ==============================
# Select features and target
# ==============================
features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]
X = data[features].copy()
y = data["Survived"]

# ==============================
# Handle missing values
# ==============================
X["Age"] = X["Age"].fillna(X["Age"].median())
X["Embarked"] = X["Embarked"].fillna(X["Embarked"].mode()[0])

# ==============================
# Encode categorical features
# ==============================
X = pd.get_dummies(X)

# ==============================
# Train-test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ==============================
# Initialize XGBoost Classifier
# ==============================
model = XGBClassifier(
    n_estimators=150, 
    max_depth=5, 
    learning_rate=0.1, 
    subsample=0.8,
    colsample_bytree=0.8, 
    random_state=42,
    eval_metric="logloss"
)

# ==============================
# Train model
# ==============================
model.fit(X_train, y_train)

# ==============================
# Make predictions
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