#

'''
compute sklearn metrics when probabilities and targets are known
'''

import pandas as pd
import os
import numpy as np

os.chdir('/Users/jessedesimone/Desktop')
df=pd.read_csv('test.csv')

#auc score
from sklearn import metrics
y_true = np.array(df['y_true'])
y_pred = np.array(df['y_pred'])
fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=1)
auc = metrics.auc(fpr, tpr)


#accuracy score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
y = np.array(df['y_true'])
pred = np.array(df['y_pred_bin'])
confusion_matrix(y, pred)
print(classification_report(y, pred))
acc=metrics.accuracy_score(y, pred)
from imblearn.metrics import sensitivity_score, specificity_score
sens=sensitivity_score(y, pred)
spec=specificity_score(y, pred)


print('AUC: ', auc); print('accuracy: ', acc); print('sensitivity: ', sens); print('specificity: ', spec)