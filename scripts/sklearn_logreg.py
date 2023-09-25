#!/usr/local/bin/python3.9
'''
logistic regression in sklearn with confusion matrix, cross validation, and auc
'''
#import packages
import warnings
import sys
sys.path.insert(0, '/Users/jessedesimone/DeSimone_Github/P01_MCI_DMRI/p01_mci_dmri') #configure system path
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

df=pd.read_csv('</path/to/infile.csv>')

# select data for initial model
cor_target = abs(corr["<correlation variable>"])       #get absolute value of the correlation
relevant_features = cor_target[cor_target>0.2]
features_f=[index for index, value in relevant_features.iteritems()]
features_f.remove('<response variable>')   #remove outcome from list of features
X=df[features_f]
y=df['<response variable>']

#get shape
X.shape, y.shape

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 4)
X_train.shape, y_train.shape
X_test.shape, y_test.shape

#create and fit StandardScaler instance
min_max_scaler = preprocessing.MinMaxScaler()
X_train_scaled = min_max_scaler.fit_transform(X_train)
X_test_scaled = min_max_scaler.transform(X_test)

#build logistic regression model
lr_mod = LogisticRegression()
lr_mod.fit(X_train_scaled, y_train)
y_predicted = lr_mod.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_predicted)
print(cm)

score = lr_mod.score(X_test_scaled, y_test)
print('Test Accuracy Score', score)

plt.clf()
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('<title>')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()

#cross validation
scores = cross_val_score(lr_mod, X_train_scaled, y_train, cv=5)
print('Cross-Validation Accuracy Scores', scores)
scores = pd.Series(scores)
min_cv = scores.min()
max_cv = scores.max()
mean_cv = scores.mean()
print('Mean cross-validation score: {:.2f}'.format(mean_cv))

#plot ROC and calculate AUC
y_pred = lr_mod.decision_function(X_test_scaled)
logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_pred)
auc_logistic = auc(logistic_fpr, logistic_tpr)
plt.figure(figsize=(5, 5), dpi=100)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)
plt.xlabel('False Positive Rate -->')
plt.ylabel('True Positive Rate -->')
plt.legend()
plt.show()

