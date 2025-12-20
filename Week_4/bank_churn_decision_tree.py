# ============================================================
# Project: Bank Customer Churn Prediction using Decision Tree
# Description: Predict whether a customer will exit the bank
# ============================================================

# ==============================
# Import Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Load dataset
# ==============================
url = "https://raw.githubusercontent.com/selva86/datasets/master/Churn_Modelling.csv"
data = pd.read_csv(url)

# ==============================
# Select useful features
# ==============================
features = ["CreditScore", "Geography", "Gender", "Age", "Balance", "NumOfProducts", "IsActiveMember", "EstimatedSalary"]
X = data[features].copy()
y = data["Exited"]

# ==============================
# One-hot encode categorical features
# ==============================
X = pd.get_dummies(X)

# ==============================
# Train/test split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ==============================
# Model
# ==============================
model = DecisionTreeClassifier(criterion='entropy', max_depth=6, class_weight='balanced')
model.fit(X_train, y_train)

# ==============================
# Predictions
# ==============================
preds = model.predict(X_test)

# ==============================
# Evaluation
# ==============================
accuracy = accuracy_score(y_test, preds)
print(f"Accuracy: {accuracy*100:.2f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, preds))  

print("\n=== Classification Report ===")
print(classification_report(y_test, preds))

# ==============================
# Visualize
# ==============================
plt.figure(figsize=(25, 12))
plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    class_names=["Not Exited", "Exited"],
    fontsize=6
)
plt.show()