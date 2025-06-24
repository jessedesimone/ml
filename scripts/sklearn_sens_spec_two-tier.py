'''
Calculated dependent sensitivity and specificity for two main endpoints
Endpoint 2 is adjusted based on the outcome of tier one
i.e., if tier 1 classification is wrong, tier 2 cannot be correct and is wrong
'''

import os
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from sklearn.metrics import confusion_matrix

os.chdir('/Users/jessedesimone/Desktop')

# Load Excel file
df = pd.read_excel("Book5.xlsx", sheet_name="Sheet1")

#-----------Tier 1-----------
# Create y_true for Tier 1 (true labels)
df['y_true_msa_psp_v_pd'] = np.where(df['GroupID'] == 1, 0, 1)
y_true = df['y_true_msa_psp_v_pd']

# Create y_pred based on predicted probabilities
y_pred = df['dmri_msa_psp_v_pd (MSA/PSP Probability)_15']

# Convert probabilities to binary predictions using 0.5 threshold
df['y_pred_msa_psp_v_pd'] = (y_pred >= 0.5).astype(int)
y_pred = df['y_pred_msa_psp_v_pd']

# Create a column to indicate if y_true and y_pred match
df['tier_1_result'] = df.apply(lambda row: 1 if row['y_true_msa_psp_v_pd'] == row['y_pred_msa_psp_v_pd'] else 0, axis=1)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calculate sensitivity and specificity
sensitivity_t1 = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_t1 = tn / (tn + fp) if (tn + fp) > 0 else 0

# Output results
print('----------Tier 1 Results----------')
print(f"Sensitivity: {sensitivity_t1:.3f}")
print(f"Specificity: {specificity_t1:.3f}")

#-----------Tier 2-----------
# Subset df to only include intended groups
df2 = df[(df['GroupID'] == 2) | (df['GroupID'] == 3)].copy()
#print(df2)
#print(df2.info())

# Create y_true for Tier 2 (true labels)
df2['y_true_psp_v_msa'] = np.where(df2['GroupID'] == 2, 0, 1)
y_true = df2['y_true_psp_v_msa']

# Create y_pred based on predicted probabilities
y_pred = df2['dmri_psp_v_msa (PSP Probability)_15']

# Convert probabilities to binary predictions using 0.5 threshold
df2['y_pred_psp_v_msa'] = (y_pred >= 0.5).astype(int)
y_pred = df2['y_pred_psp_v_msa']

# Create a column to indicate if y_true and y_pred match
df2['tier_2_result'] = df2.apply(lambda row: 1 if row['y_true_psp_v_msa'] == row['y_pred_psp_v_msa'] else 0, axis=1)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# Calculate sensitivity and specificity
sensitivity_t2 = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_t2 = tn / (tn + fp) if (tn + fp) > 0 else 0

# Output results
print('----------Tier 2 Results----------')
print(f"Sensitivity: {sensitivity_t2:.3f}")
print(f"Specificity: {specificity_t2:.3f}")

#-----------Tier 2 Adjusted-----------
# Make a copy of Tier 2 predicted values
df2['y_pred_psp_v_msa_adjusted'] = df2['y_pred_psp_v_msa']

# Force incorrect prediction if Tier 1 was wrong
def adjust_prediction(row):
    if row['tier_1_result'] == 0:
        # Flip prediction so it's guaranteed to be wrong
        return 1 - row['y_true_psp_v_msa']
    else:
        return row['y_pred_psp_v_msa']

df2['y_pred_psp_v_msa_adjusted'] = df2.apply(adjust_prediction, axis=1)

# Recalculate confusion matrix with adjusted predictions
y_true = df2['y_true_psp_v_msa']
y_pred_adj = df2['y_pred_psp_v_msa_adjusted']

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_adj).ravel()

sensitivity_t2_adj = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity_t2_adj = tn / (tn + fp) if (tn + fp) > 0 else 0

# Output
print('----------Tier 2 Adjusted Results----------')
print(f"Adjusted Sensitivity: {sensitivity_t2_adj:.3f}")
print(f"Adjusted Specificity: {specificity_t2_adj:.3f}")

# Create DataFrame with desired structure
metrics_df = pd.DataFrame({
    'Tier 1': [sensitivity_t1, specificity_t1],
    'Tier 2': [sensitivity_t2, specificity_t2],
    'Tier 2_adj': [sensitivity_t2_adj, specificity_t2_adj],
}, index=['Sensitivity', 'Specificity'])

# Write output to excel file
with pd.ExcelWriter('sens_spec_output.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Tier_1_df', index=False)
    df2.to_excel(writer, sheet_name='Tier_2_df', index=False)
    metrics_df.to_excel(writer, sheet_name='metrics')


