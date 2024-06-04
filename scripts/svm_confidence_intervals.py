#!/usr/local/bin/python3.9
'''
module for calculating confidence intervals for metrics from 50 simulations of an SVM model
module will loop through columns in input file =(e.g., AUC, sens, spec, ppv, npv) and calculate confidence interval for each column of values
output will be provided as a dataframe
'''

# import packages
import os
import numpy as np
from scipy import stats
import pandas as pd

path_to_file = '/Users/jessedesimone/Desktop'
file = 'CI_infile_pd_msa.csv'
infile = os.path.join(path_to_file,file)
data = pd.read_csv(infile)

# set confidence level
conf = 0.95

list_of_cols = list(data.columns)
list_of_mean=[]
list_of_std=[]
list_of_low_ci=[]
list_of_high_ci=[]
for col in data.columns[0:]:
    #print('''Metric - ''', col)
    # Calculate mean and standard deviation
    col_mean=data[col].mean()
    #print('Mean: ', col_mean)
    list_of_mean.append(col_mean)
    col_std = data[col].std()  # Use ddof=1 for sample standard deviation
    #print('SD: ', col_std)
    list_of_std.append(col_std)
    
    # Degrees of freedom
    n = len(col)
    df = n - 1
    
    # Desired confidence level
    confidence_level = conf
    
    # t-critical value for 95% confidence interval
    t_critical = stats.t.ppf(1 - (1 - confidence_level) / 2, df)
    #print('T Critical Value: ', t_critical)
    
    # Margin of error
    margin_of_error = t_critical * (col_std / np.sqrt(n))
    #print('Error Margin: ', margin_of_error)
    
    # Confidence interval
    ci_low = round(col_mean - margin_of_error, 3)
    list_of_low_ci.append(ci_low)
    ci_high = round(col_mean + margin_of_error, 3)
    list_of_high_ci.append(ci_high)
    confidence_interval = (col_mean - margin_of_error, col_mean + margin_of_error)
    #print('Confidence Interval at ', confidence_level, ': ', confidence_interval)
    
# Append results to dataframe and output as csv
d = {'Mean':list_of_mean,
     'SD':list_of_std,
     '95% Lower':list_of_low_ci,
     '95% Upper':list_of_high_ci
     }
final_df=pd.DataFrame(d, index=list_of_cols)
print(final_df)
