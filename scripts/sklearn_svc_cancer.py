#!/usr/local/bin/python3.9
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# Load the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into train and test sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Support Vector Machine classifier with C=1 and linear kernel
svm_classifier = SVC(C=1, kernel='linear')

# Train the SVM classifier
svm_classifier.fit(X_train, y_train)

# Predict on the test set
y_pred = svm_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Predict probabilities for the test set
y_proba = svm_classifier.decision_function(X_test)

# Calculate the ROC AUC score for the test set
roc_auc = roc_auc_score(y_test, y_proba)

# Calculate sensitivity and specificity
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# Print final metrics on test set
print("------Final Test Metrics------")
print("ROC AUC Score:", roc_auc)
print("Sensitivity:", sensitivity)
print("Specificity:", specificity)
print("Accuracy:", accuracy)