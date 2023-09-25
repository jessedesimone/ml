#!/usr/local/bin/python3.6

'''
module for train test split to prepare data for aidp svm 
'''
#import required packages
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

#define paths
top = '/Users/jessedesimone/DeSimone_Github/neuropacs/aiddml'
data_dir = os.path.join(top + '/data')
out_dir = os.path.join(top + '/input')

#read data
df = pd.read_excel(os.path.join(data_dir + '/test.xlsx'))

#define predictors (X) and outcome (y) variables


#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)