#!/usr/local/bin/python

'''
compute sklearn metrics when probabilities and targets are known
'''

import pandas as pd
import os
import numpy as np
import logging
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from  datetime import datetime
import matplotlib.pyplot as plt

#define data directories
class directories:
    '''Should be accessible in each module'''
    top_dir = '/Users/jessedesimone/Desktop'
    outfile_dir = os.path.join(top_dir, 'ml_output/')

os.chdir(directories.top_dir)

#create required directories
dirs = [directories.outfile_dir]
for dir in dirs: 
    dir_exist=os.path.exists(dir)
    if not dir_exist:
        os.makedirs(dir)

#confusion matrix
def conmatscores(y_true, y_pred):
    y_pred_class = np.array(np.round(y_pred))
    cm = confusion_matrix(y_true, y_pred_class) #assume y_pred is fed in as probabilities
    #print(cm)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total=(tn+fp+fn+tp)
    accuracy = (tp+tn)/(total)
    sens = tp / (tp+fn)
    spec = tn / (fp+tn)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, sens, spec, ppv, npv, auc

# plot roc curve for train or test
def plotroc_single(Y_true, Y_pred, title: str, outname: str, type: str):
    fpr, tpr, thresh = roc_curve(Y_true, Y_pred)
    auc = round(roc_auc_score(Y_true, Y_pred), 4)
    if type == 'train':
        lab='train'
    elif type == 'test':
        lab='test'
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=lab + ' ' + 'AUC=' + str(auc))
    plt.plot([0,1], [0,1], 'k--')
    plt.title(title + ' ' + type, color='k')
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.legend(loc=4, frameon=False)
    plt.tight_layout()
    plt.savefig(directories.outfile_dir + outname + '_' + type + '.jpg', dpi=300)
    plt.show()

#dataframe with known true targets and predicted probabilities
df=pd.read_csv('test.csv')

#define classes
target = np.array(df['y_true'])
pred = np.array(df['y_pred'])

#get auc score only
# from sklearn import metrics
# fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
# auc = metrics.auc(fpr, tpr); print('AUC = ', round(auc,4))

test=conmatscores(target, pred)
metric_list=['accuracy', 'sens', 'spec', 'ppv', 'npv', 'auc']
score_list=[test[0], test[1], test[2], test[3], test[4], test[5]]
dict = {'metric': metric_list, 'score': score_list} 
df_out = pd.DataFrame(dict)
print(df_out)
df_out.to_csv(directories.outfile_dir + 'metrics.csv', index=False)

#plot ROC curve
plotroc_single(target, pred, 'PD vs. atypical Parkinsonism', 'pd_v_spd', 'test')