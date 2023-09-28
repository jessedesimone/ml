#!/usr/local/bin/python3.9

'''
module for train test split to prepare data for aidp svm 
'''
#import required packages
import os;
import pandas as pd;
import numpy as np;
from sklearn.model_selection import train_test_split;

#define paths
top = '/Users/jessedesimone/desimone_github/ml/aidp'
#data_dir = os.path.join(top + '/infiles')
data_dir = os.path.join(top + '/tests/resources')
os.chdir(data_dir)

#read data
df = pd.read_excel('1002_Data_no_Subj_Site.xlsx')

#train test split | save train and test data as outfiles
train, test = train_test_split(df, test_size=0.2, random_state=42)
print(train.shape)
print(test.shape)
train.to_excel('train_set.xlsx', index=False)
test.to_excel('test_set.xlsx', index=False)