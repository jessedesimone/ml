import os
import pandas as pd
from sklearn.metrics import confusion_matrix

os.chdir('/Users/jessedesimone/Desktop')

# Load Excel file
df = pd.read_excel("input.xlsx", sheet_name="Sheet1")

# Extract true labels and predicted probabilities
y_true = df['y_true']
y_pred = df['y_pred']

# Convert probabilities to binary predictions using 0.5 threshold
y_pred = (y_pred >= 0.5).astype(int)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calculate sensitivity and specificity
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Output results
print(f"Sensitivity: {sensitivity:.3f}")
print(f"Specificity: {specificity:.3f}")
