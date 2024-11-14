#!/usr/local/bin/python

'''
compute sklearn metrics when probabilities and targets are known
'''
# Import packages
import os
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import numpy as np

# Define data directories
class directories:
    '''Should be accessible in each module'''
    top_dir = '/Users/jessedesimone/Desktop'
    outfile_dir = os.path.join(top_dir, 'ml_output/')

os.chdir(directories.top_dir)

# Create output directory
dirs = [directories.outfile_dir]
for dir in dirs: 
    dir_exist=os.path.exists(dir)
    if not dir_exist:
        os.makedirs(dir)

# Import dataframe
df = pd.read_excel('probabilities.xlsx')

'''remove subject column'''
if 'Subject' in df.columns:    
    df = df.drop(columns=['Subject'])

'''create GroupLabel for PD/APD'''
df['GroupLabel'] = np.where(df['GroupID'] == 1, 'PD', 'APD')        #create PD/APD column

# Subsetting dataframe
'''apd_vs_pd'''
df_apd_pd = df[['GroupID','dmri_msa_psp_v_pd (MSA/PSP Probability)']].copy()
df_apd_pd['y_pred'] = df_apd_pd['dmri_msa_psp_v_pd (MSA/PSP Probability)']
df_apd_pd['y_true'] = df_apd_pd['GroupID'].apply(lambda x: 0 if x == 1 else 1)
'''psp_v_msa'''
df_psp_msa = df[df['GroupID'].isin([2, 3])]
df_psp_msa = df_psp_msa[['GroupID','dmri_psp_v_msa (PSP Probability)']].copy()
df_psp_msa['y_pred'] = df_psp_msa['dmri_psp_v_msa (PSP Probability)']
df_psp_msa['y_true'] = df_psp_msa['GroupID'].apply(lambda x: 0 if x == 2 else 1)
'''pd_v_msa'''
df_pd_msa = df[df['GroupID'].isin([1, 2])]
df_pd_msa = df_pd_msa[['GroupID','dmri_pd_v_msa (PD Probability)']].copy()
df_pd_msa['y_pred'] = df_pd_msa['dmri_pd_v_msa (PD Probability)']
df_pd_msa['y_true'] = df_pd_msa['GroupID'].apply(lambda x: 0 if x == 2 else 1)
'''pd_v_psp'''
df_pd_psp = df[df['GroupID'].isin([1, 3])]
df_pd_psp = df_pd_psp[['GroupID','dmri_pd_v_psp (PD Probability)']].copy()
df_pd_psp['y_pred'] = df_pd_psp['dmri_pd_v_psp (PD Probability)']
df_pd_psp['y_true'] = df_pd_psp['GroupID'].apply(lambda x: 0 if x == 3 else 1)

dataframes = {'df1': df_apd_pd, 'df2': df_psp_msa, 'df3': df_pd_msa, 'df4': df_pd_psp}

# Function for confusion matrix
def calculate_metrics(y_true, y_pred, threshold=0.5):
    # Calculate AUC with continuous y_pred
    auc = roc_auc_score(y_true, y_pred)
    
    # Convert y_pred to binary based on the threshold
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Confusion matrix elements
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    # Calculate sensitivity, specificity, PPV, NPV, and accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    return {
        'AUC': auc,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'PPV': ppv,
        'NPV': npv,
        'Accuracy': accuracy
    }

# Create a dictionary to store the metrics for each dataframe
metrics_results = {}

# Calculate metrics for each dataframe and store results
for name, df in dataframes.items():
    metrics = calculate_metrics(df['y_true'], df['y_pred'])
    metrics_results[name] = pd.DataFrame(metrics, index=[0])

# Write the results to an Excel file with separate sheets for each dataframe
output_file = os.path.join(directories.outfile_dir, 'metrics_results.xlsx')
with pd.ExcelWriter(output_file) as writer:
    for name, result_df in metrics_results.items():
        result_df.to_excel(writer, sheet_name=name, index=False)