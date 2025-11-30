# ---------------------------------------------------------
# PROJECT: Model Selection + Bias/Variance + Error Analysis
# ---------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Load Dataset
# Dataset: 'spam.csv' (text + label)
df = pd.read_csv('Week_3/ML_Pipeline_Diagnostic/spam.csv')

# Features and labels
X = df['text']
y = df['label']

# Train / CV / Test Split
# 60% train, 20% cv, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)

X_cv, X_test, y_cv, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# Convert Text to Vectors
vectorizer = CountVectorizer()

# Fit only on training data
vectorizer.fit(X_train)

X_train_vec = vectorizer.transform(X_train)
X_cv_vec = vectorizer.transform(X_cv)
X_test_vec = vectorizer.transform(X_test)

# Train multiple models
log_reg = LogisticRegression(max_iter=2000)
svm = SVC()
nb = MultinomialNB()

log_reg.fit(X_train_vec, y_train)
svm.fit(X_train_vec, y_train)
nb.fit(X_train_vec, y_train)

# Evaluate on CV set (model selection)
pred_lr = log_reg.predict(X_cv_vec)
pred_svm = svm.predict(X_cv_vec)
pred_nb = nb.predict(X_cv_vec)

acc_lr = accuracy_score(y_cv, pred_lr)
acc_svm = accuracy_score(y_cv, pred_svm)
acc_nb = accuracy_score(y_cv, pred_nb)

print("CV Accuracy (Logistic Regression):", acc_lr)
print("CV Accuracy (SVM):", acc_svm)
print("CV Accuracy (Naive Bias):", acc_nb)

best_model_name = max(
    [("Logistic Regression", acc_lr),
     ("SVM", acc_svm),
     ("Naive Bias", acc_nb)],
     key=lambda x: x[1]
)[0]

print("\nBest Model on CV set:", best_model_name)

# Bias / Variance Diagnosis
# Compare training accuracy vs CV accuracy
# If train >> cv → high variance
# If both low → high bias
# If both high → good fit
train_acc_lr = accuracy_score(y_train, log_reg.predict(X_train_vec))
train_acc_svm = accuracy_score(y_train, svm.predict(X_train_vec))
train_acc_nb = accuracy_score(y_train, nb.predict(X_train_vec))

print("\nTrain vs CV Accuracy:")

print(f"Logistic Regression: Train={train_acc_lr:.3f}, CV={acc_lr:.3f}")
print(f"SVM: Train={train_acc_svm:.3f}, CV={acc_svm:.3f}")
print(f"Naive Bias: Train={train_acc_nb:.3f}, CV={acc_nb:.3f}")

# Plot Learning Curve (Logistic Regression as example)
train_sizes = [100, 300, 600, 1000, len(X_train)]

train_scores = []
cv_scores = []

for size in train_sizes:
  x_part = X_train_vec[:size]
  y_part = y_train[:size]

  model = LogisticRegression(max_iter=2000)
  model.fit(x_part, y_part)

  train_scores.append(accuracy_score(y_part, model.predict(x_part)))
  cv_scores.append(accuracy_score(y_cv, model.predict(X_cv_vec)))

plt.plot(train_sizes, train_scores, label="Train Accuracy")
plt.plot(train_sizes, cv_scores, label="CV Accuracy")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# Error Analysis (on CV set)
if best_model_name == "Logistic Regression":
  model = log_reg
elif best_model_name == "SVM":
  model = svm
else:
  model = nb

cv_predictions = model.predict(X_cv_vec)

missclassified = X_cv[cv_predictions != y_cv]
print("\nExamples of Missclassifed Messages:")
print(missclassified.head(10))

# Decide What to Try Next
#   - Is the model suffering from bias or variance?
#   - Would more data help?
#   - Would regularization help?
#   - Would feature engineering help?
#   - Which error category is biggest?
#   - What is your next action?
