# ============================================================
# Project: Breast Cancer Classification using Decision Tree
# Description: ML model to classify tumors as benign or malignant
# ============================================================

# ==============================
# Import Libraries
# ==============================
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# ==============================
# Load data
# ==============================
data = load_breast_cancer()
X = data.data
y = data.target

# ==============================
# Split into train/test
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion='gini', max_depth=3)
model.fit(X_train, y_train)

# ==============================
# Predict
# ==============================
predictions = model.predict(X_test)

# ==============================
# Accuracy
# ==============================
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy*100:.2f}%")

# ==============================
# Confusion Matrix
# ==============================
print("\n=== Confuion Matrix ===")
cm = confusion_matrix(y_test, predictions)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot()
plt.show()

# ==============================
# Classification Report
# ==============================
print("\n=== Classification Report ===")
print(classification_report(y_test, predictions))

# ==============================
# Visualize Decision Tree
# ==============================
plt.figure(figsize=(14, 6))
plot_tree(
    model, 
    filled=True, 
    feature_names=data.feature_names, 
    class_names=data.target_names, 
    fontsize=8
)
plt.show()


