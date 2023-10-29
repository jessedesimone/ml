#!/usr/local/bin/python

'''
Module for plotting ROC curves on train and test predictions output from AIDP SVM
'''

#================== configuration ==================

#import packages
import os
import logging
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

'''note: must use sklearn version 0.21.3 or lower for handling pkl files
pkl files contain the optomized parameters from aidp svm model training
pkl files are needed here
should be using Python 3.6.13 |Anaconda, Inc.| scikit-learn==0.21.3
'''
#version control
# skvers = sklearn.__version__
# #NEED to use the same ML conda version (scikit-learn==0.19.0)
# if skvers != "0.21.3" :
#     !pip install scikit-learn==0.19.0
#     import sklearn

#define data directories
class directories:
    '''Should be accessible in each module'''
    top_dir = '/Users/jessedesimone/desimone_github'
    script_dir = os.path.join(top_dir, 'ml/scripts/')
    log_dir =  os.path.join(top_dir, 'ml/logs/')
    proj_dir = os.path.join(top_dir, 'ml/aidp/') 
    infile_dir = os.path.join(proj_dir, 'resources/models/model_231027/')
    outfile_dir = os.path.join(top_dir, 'ml/mlreports/')

#configure logger
os.chdir(directories.script_dir)
from logger import configure_logger
configure_logger(directories.log_dir)
logger = logging.getLogger(__name__)
logger.info("Starting plot_roc_aidp.py")

os.chdir(directories.infile_dir)
logger.info("Infile directory: {}".format(directories.infile_dir))
logger.info("Current kernel is " + os.environ['CONDA_DEFAULT_ENV'])

#load Dataframes
logger.info("Loading dataframes")
#training model
logger.info("Loading training model")
df_tr=pd.read_excel(directories.infile_dir +'aidd_training_model_231027.xlsx', header=0, index_col="Subject")
df_tr=df_tr.drop(columns=['Unnamed: 0'], axis=1)    #drop old index column
df_tr_Y=df_tr[['GroupID']]    #set GroupID as true Y
df_tr_X=df_tr.drop(['GroupID', 'site', 'scanner'], axis=1)      #set X data

#testing model
logger.info("Loading testing model")
df_te=pd.read_excel(directories.infile_dir +'aidd_testing_model_231027.xlsx', header=0, index_col="Subject");
df_te=df_te.drop(columns=['Unnamed: 0'], axis=1)    #drop old index column
df_te_Y=df_te[['GroupID']]    #set GroupID as true Y
df_te_X=df_te.drop(['GroupID', 'site', 'scanner'], axis=1)      #set X data

#load pkl files (aidp svm optomized models)
mod_advdlb=pickle.load(open(directories.infile_dir + 'dmri/' + 'ad_v_dlb.pkl', 'rb'))
mod_ftdvad=pickle.load(open(directories.infile_dir + 'dmri/' + 'ftd_v_ad.pkl', 'rb'))
mod_ftdvdlb=pickle.load(open(directories.infile_dir + 'dmri/' + 'ftd_v_dlb.pkl', 'rb'))
model_list = [mod_advdlb, mod_ftdvad, mod_ftdvdlb]

#Dictionary for label redefining (1 for 'positive' class (higher probability), 0 for 'negative' class)
'''
Original values
AD=1
DLB=2
CON=3
FTD=4
'''
#'''ad pos class dict'''
ad_v_con_dict = {"GroupID": {3:0}}      #AD pos class = 1; CON neg class = 0
ad_v_dlb_dict = {"GroupID": {2:0}}      #AD pos class = 1, DLB neg class = 0
ad_v_ftd_dict = {"GroupID": {4:0}}      #AD pos class = 1; FTD neg class = 0
ad_v_dem_dict = {"GroupID": {2:0, 4:0}}      #AD pos class = 1; DLB/FTD neg class = 0
ad_v_all_dict = {"GroupID": {2:0, 3:0, 4:0}}      #AD pos class = 1; DLB/CON/FTD neg class = 0
#'''dlb pos class dict'''
dlb_v_con_dict = {"GroupID": {2:1, 3:0}}      #DLB pos class = 1; CON neg class = 0
dlb_v_ad_dict = {"GroupID": {2:1, 1:0}}      #DLB pos class = 1, AD neg class = 0
dlb_v_ftd_dict = {"GroupID": {2:1, 4:0}}      # DLB pos class = 1; FTD neg class = 0
dlb_v_dem_dict = {"GroupID": {2:1, 1:0, 4:0}}      #DLB pos class = 1; AD/FTD neg class = 0
dlb_v_all_dict = {"GroupID": {2:1, 1:0, 3:0, 4:0}}      #DLB pos class = 1; AD/CON/FTD neg class = 0
#'''ftd pos class dict'''
ftd_v_con_dict = {"GroupID": {4:1, 3:0}}      #FTD pos class = 1; CON neg class = 0
ftd_v_ad_dict = {"GroupID": {4:1, 1:0}}      #FTD pos class = 1, AD neg class = 0
ftd_v_dlb_dict = {"GroupID": {4:1, 2:0}}      #FTD pos class = 1; DLB neg class = 0
ftd_v_dem_dict = {"GroupID": {4:1, 1:0, 2:0}}      #FTD pos class = 1; AD/DLB neg class = 0
ftd_v_all_dict = {"GroupID": {4:1, 1:0, 2:0, 3:0}}      #FTD pos class = 1; AD/DLB/CON neg class = 0
#'''extras'''
#dvb_dict = {"GroupID": {1:"A", 3:"A", 2:"B"}}
#cvb_dict = {"GroupID": {1:"A", 2:"A", 3:"B"}}
#tf_dict = {"GroupID": {"B":1, "A":0}}

#create dict list to call later
dict_list = [ad_v_con_dict, ad_v_dlb_dict, ad_v_ftd_dict, ad_v_dem_dict, ad_v_all_dict,
             dlb_v_con_dict, dlb_v_ad_dict, dlb_v_ftd_dict, dlb_v_dem_dict, dlb_v_all_dict,
             ftd_v_con_dict, ftd_v_ad_dict, ftd_v_dlb_dict, ftd_v_dem_dict, ftd_v_all_dict]

#define functions 
# roc curve function 
def plotroc(trfp, trtp, trauc, tefp, tetp, teauc, title: str, outname: str):
    '''trfp/tefp = training false positive rate / testing
    trtp/tetp = training true positive rate / testing
    trauc/teauc = training auc / testing
    title = figure title
    outname = output file name (no extension)'''
    print('plotting ROC curve')
    fig=plt.figure()
    plt.plot(trfp, trtp, label="Training AUC=%0.3f" %trauc)
    plt.plot(tefp, tetp, label="Testing AUC=%0.3f" %teauc)
    plt.plot([0,1], [0,1], 'k--')
    plt.title(title, color='k', rotation='vertical', x=-.15, y=.345)
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.legend(loc=4, frameon=False)
    plt.savefig(outname +'.jpg', dpi=300)
    plt.show()
    return fig








#================== plot ROC curves ==================
logger.info("Plotting ROC curves")
#AD vs DLB
logger.info("AD vs DLB")
dxmodel = model_list[0]       #load classification model
'''select only rows with AD and DLB cases; then rename based on data_dict'''
df_tr_Y_addlb = df_tr_Y.loc[(df_tr_Y.GroupID == 1) & (df_tr_Y.GroupID == 2)]