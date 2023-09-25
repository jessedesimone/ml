#!/usr/bin/env python3

'''
'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler

#import dataframe
df=pd.read_csv('<path/to/infile.csv>')

#standard scaler transform
scaler = preprocessing.StandardScaler()
ss_df = scaler.fit_transform(df)
ss_df = pd.DataFrame(ss_df, columns = df.columns)

#robust scaler transform
scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(df)
robust_df = pd.DataFrame(robust_df, columns = df.columns)

#minmax scaler transform
scaler = preprocessing.MinMaxScaler()
minmax_df = scaler.fit_transform(df)
minmax_df = pd.DataFrame(minmax_df, columns = df.columns)

#plot transformed df
fig, axes = plt.subplots(ncols=len(robust_df.columns), figsize=(10,5))
for ax, col in zip(axes, robust_df.columns):
  sns.histplot(robust_df[col], ax=ax, kde=True)
plt.tight_layout()
plt.show()