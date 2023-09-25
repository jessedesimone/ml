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
top = '/Users/jessedesimone/desimone_github/ml/aidp'
data_dir = os.path.join(top + '/infiles')
os.chdir(data_dir)

#read data
df = pd.read_excel('data_ad_dlb_ftd_full.xlsx')

#train test split | save train and test data as outfiles
train, test = train_test_split(df, test_size=0.2, random_state=42)
print(train.shape)
print(test.shape)
train.to_excel('data_ad_dlb_ftd_train.xlsx', index=False)
test.to_excel('data_ad_dlb_ftd_test.xlsx', index=False)