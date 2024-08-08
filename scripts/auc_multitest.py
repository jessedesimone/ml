#!/usr/local/bin/python

'''
Module for multiple comparison corrections of AUC values obtained in SVM-C analysis
Use either DeLong's test or Permutation test to compute p-value on the observed AUC
Then use FDR to correct for multiple p values

This code assumes you have a single excel file with multiple sheets
Each sheet should contain the y_true and y_pred for a single endpoint (e.g., PD vs APD)
'''
# Import packages
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
#from mlxtend.evaluate import delong_roc_variance

# Function to compute the variance of the AUC using bootstrapping
def bootstrap_auc_variance(y_true, y_scores, n_iterations=1000):
    aucs = np.zeros(n_iterations)
    n = len(y_true)
    for i in range(n_iterations):
        indices = np.random.choice(n, n, replace=True)
        y_true_boot = y_true[indices]
        y_scores_boot = y_scores[indices]
        aucs[i] = roc_auc_score(y_true_boot, y_scores_boot)
    return np.var(aucs)

# Function to perform DeLong's Test
def delong_roc_test(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    auc_var = bootstrap_auc_variance(y_true, y_scores)
    z = (auc - 0.5) / np.sqrt(auc_var)
    p_value = 2 * norm.cdf(-abs(z))
    return auc, p_value

# File path to the Excel file
file_path = '/Users/jessedesimone/Desktop/Book3.xlsx'  # Replace with your file path

# Read all sheets from the Excel file
xls = pd.ExcelFile(file_path)
sheet_names = xls.sheet_names

# Initialize lists to store p-values
p_values_delong = []
p_values_permutation = []

# Number of permutations for permutation test
n_permutations = 1000

# Loop through each sheet
for sheet_name in sheet_names:
    # Read the data from the sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Extract y_true and y_pred columns
    y_true = df['y_true'].values
    y_scores = df['y_pred'].values
    
    # Compute DeLong's Test p-value
    _, p_value_delong = delong_roc_test(y_true, y_scores)
    p_values_delong.append(p_value_delong)
    
    # Compute permutation test p-value
    observed_auc = roc_auc_score(y_true, y_scores)
    permuted_aucs = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        y_true_permuted = np.random.permutation(y_true)
        permuted_aucs[i] = roc_auc_score(y_true_permuted, y_scores)
    
    p_value_permutation = np.mean(permuted_aucs >= observed_auc)
    p_values_permutation.append(p_value_permutation)

# Correct for multiple comparisons using FDR
p_values_corrected_delong = multipletests(p_values_delong, method='fdr_bh')[1]
p_values_corrected_permutation = multipletests(p_values_permutation, method='fdr_bh')[1]

# Print results with 5 decimal places
print("Original p-values (DeLong's Test):", [f"{p:.5f}" for p in p_values_delong])
print("FDR-corrected p-values (DeLong's Test):", [f"{p:.5f}" for p in p_values_corrected_delong])

print("Original p-values (Permutation Test):", [f"{p:.5f}" for p in p_values_permutation])
print("FDR-corrected p-values (Permutation Test):", [f"{p:.5f}" for p in p_values_corrected_permutation])
