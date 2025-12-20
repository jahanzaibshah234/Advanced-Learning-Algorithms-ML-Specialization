# ============================================================
# Project: Wine Quality Prediction using Decision Tree Classifier
# Description: Predict if a wine is of good quality
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
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep=";")

# ==============================
# Convert quality to binary (good/bad)
# ==============================
data["QualityLabel"] = data["quality"].apply(lambda x: 1 if x >= 6 else 0)

# ==============================
# Split features and target
# ==============================
X = data.drop(["quality", "QualityLabel"], axis=1)
y = data["QualityLabel"]

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
plt.figure(figsize=(25, 10))
plot_tree(
    model,
    filled=True,
    feature_names=X.columns,
    class_names=["Bad Wine", "Good Wine"],
    fontsize=6
)
plt.show()
