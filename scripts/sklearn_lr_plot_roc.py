#!/usr/local/bin/python3.9

'''script for sklearn logistic regression and plotting ROC curve'''

# import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib import rcParams

#read data and define variables
df = pd.read_csv('/Users/jessedesimone/Desktop/youden.csv')
X = df.iloc[:, 0].values        #select column 0
y = df.iloc[:, 1].values        #select column 1
 
#split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
X_train=X_train.reshape(-1,1)
X_test=X_test.reshape(-1,1)
 
#build model
model = LogisticRegression()
model.fit(X_train, y_train)
 
#predict probabilities
probs = model.predict_proba(X_test)
 
#keeping only positive class
probs = probs[:, 1]
 
#calculate fpr and tpr, auc score
fpr, tpr, thresholds = roc_curve(y_test, probs)
auc_lr=roc_auc_score(y_test, probs)
 
#Plotting the figure
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False
plt.figure(figsize = (10,6))
plt.plot(fpr, tpr, color='red', label=f'AUC (Logistic Regression) = {auc_lr:.3f}')
#plt.plot(fpr2, tpr2, color='red', label=f'AUC (Logistic Regression) = {auc_lr2:.3f}')    #add second line; need to define new terms a priori
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--', label="Baseline")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic Curve')
plt.legend(loc='lower right')
plt.show()