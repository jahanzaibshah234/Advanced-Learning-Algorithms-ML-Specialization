# ============================================================
# Project: Titanic Survival Prediction using Decision Tree
# Description: Predict passenger survival using Decision Tree Classifier
# ============================================================

# ==============================
# Import Libraries
# ==============================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree, DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================
# Load dataset
# ==============================
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# ==============================
# Select features
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
# One-Hot Encoding
# ==============================
X = pd.get_dummies(X)

# ==============================
# Split into train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# ==============================
# Model
# ==============================
model = DecisionTreeClassifier(criterion='entropy', max_depth=6)
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
# Visualize
# ==============================
plt.figure(figsize=(25, 12))
plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    class_names=["Not Survived", "Survived"],
    fontsize=6
)

plt.show()
