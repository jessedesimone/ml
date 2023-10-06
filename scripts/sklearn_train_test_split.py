#!/usr/local/bin/python3.9

'''
module for train test split to prepare data for aidp svm 
'''
#import required packages
import os;
import pandas as pd;
import numpy as np;
from sklearn.model_selection import train_test_split;
import sys

#define paths
top = '/Users/jessedesimone/desimone_github/ml/aidp'
data_dir = os.path.join(top + '/infiles')
#data_dir = os.path.join(top + '/tests/resources')
os.chdir(data_dir)

class FileError(OSError):
    def __init__(self, filename):
        self.filename = filename
        
    def __str__(self):
        return f"++ Error in accessing file: {self.filename}\n ++ Check path"

#read data
infile='hello.xlsx'
print('++ reading data')
try:
    df = pd.read_excel(infile)
    print('++ File found')
except Exception as e:
    print (FileError(infile),e)
    sys.exit()

#train test split | save train and test data as outfiles
print('++ Running train test split')
train, test = train_test_split(df, test_size=0.2, random_state=42)
print(train.shape)
print(test.shape)
train.to_excel('aidd_training.xlsx', index=False)
test.to_excel('aidd_testing.xlsx', index=False)
print('++ train/test split done')