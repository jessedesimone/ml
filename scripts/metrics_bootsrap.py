#!/usr/local/bin/python3.9

'''
Module for:
confusion matrix
bootsrapping metrics
confidence intervals

'''
#import packages
import os
import logging
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import pickle
import numpy as np
import sys

#version control
# skvers = sklearn.__version__
# #NEED to use the same ML conda version (scikit0learn==0.19.0)
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

#create required directories
dirs = [directories.log_dir, directories.outfile_dir]
for dir in dirs: 
    dir_exist=os.path.exists(dir)
    if not dir_exist:
        os.makedirs(dir)

#configure logger
os.chdir(directories.script_dir)
from logger import configure_logger
configure_logger(directories.log_dir)
logger = logging.getLogger(__name__)
logger.info("Starting bootstrap code")

os.chdir(directories.infile_dir)
logger.info("Infile directory: {}".format(directories.infile_dir))
logger.info("Current kernel is " + os.environ['CONDA_DEFAULT_ENV'])

#Load Dataframes
logger.info("Loading dataframes")
df_tr=pd.read_excel(directories.infile_dir +'aidd_training_model_231027.xlsx', header=0, index_col="Subject")
logger.info("training model")
logger.info(df_tr.head())
df_te=pd.read_excel(directories.infile_dir +'aidd_testing_model_231027.xlsx', header=0, index_col="Subject")
logger.info("testing model")
logger.info(df_te.head())
keep_columns=[
    "GroupID",
    "dmri_ad_v_dlb (AD Probability)",
    "dmri_ad_v_con (AD Probability)",
    "dmri_ad_v_all (AD Probability)",
    "dmri_dlb_v_con (DLB Probability)",
    "dmri_dlb_v_all (DLB Probability)",
    "dmri_con_v_dem (CON Probability)",
    "dmri_ftd_v_ad (FTD Probability)",
    "dmri_ftd_v_all (FTD Probability)",
    "dmri_ftd_v_con (FTD Probability)",
    "dmri_ftd_v_dlb (FTD Probability)"
    ]
df_tr_trim=df_tr[keep_columns]
df_te_trim=df_te[keep_columns]

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

dict_list = [ad_v_con_dict, ad_v_dlb_dict, ad_v_ftd_dict, ad_v_dem_dict, ad_v_all_dict,
             dlb_v_con_dict, dlb_v_ad_dict, dlb_v_ftd_dict, dlb_v_dem_dict, dlb_v_all_dict,
             ftd_v_con_dict, ftd_v_ad_dict, ftd_v_dlb_dict, ftd_v_dem_dict, ftd_v_all_dict]

#define functions
def conmatscores(y_true, y_pred):
    y_pred_class = np.array(np.round(y_pred))
    cm = confusion_matrix(y_true, y_pred_class) #assume y_pred is fed in as probabilities
    # print(cm)
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    # tn, fp, fn, tp = cm([0,1,0,1], [1,1,1,0]).ravel()
    total=(tn+fp+fn+tp)
    accuracy = (tp+tn)/(total)
    sens = tp / (tp+fn)
    spec = tn / (fp+tn)
    ppv = tp / (tp+fp)
    npv = tn / (tn+fn)
    auc = roc_auc_score(y_true, y_pred)
    return accuracy, sens, spec, ppv, npv, auc

def bootstrapper(Y_true, Y_pred, savename: str, r_seed=42, n_bootstraps=1000):
    acc_bs_scores = []
    sens_bs_scores = []
    spec_bs_scores = []
    ppv_bs_scores = []
    npv_bs_scores = []
    auc_bs_scores = []
    rng = np.random.RandomState(r_seed)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(Y_pred), len(Y_pred))
        if len(np.unique(Y_true[indices])) < 2:
            continue
        acc, sens, spec, ppv, npv, auc = conmatscores(Y_true[indices], Y_pred[indices])
        acc_bs_scores.append(acc)
        sens_bs_scores.append(sens)
        spec_bs_scores.append(spec)
        ppv_bs_scores.append(ppv)
        npv_bs_scores.append(npv)
        auc_bs_scores.append(auc)
    df_report = pd.DataFrame( {"Accuracy": acc_bs_scores, "Sensitivity": sens_bs_scores, "Specificity": spec_bs_scores,
                               "PPV": ppv_bs_scores, "NPV": npv_bs_scores, "AUC": auc})
    df_report.to_excel(directories.outfile_dir + savename + '_bootstrap_report.xlsx', index=0)

def report_mean_ci(dfname: str):
    df = pd.read_excel(directories.outfile_dir + dfname + '_bootstrap_report.xlsx')
    df_sum=pd.DataFrame(index=df.columns, columns=["Mean", "Lower", "Upper"])
    for metric in df.columns:
        metric_array = np.array(df[metric])
        metric_array.sort()
        mean_met = np.mean(metric_array)
        lower_met = metric_array[int(.025*len(metric_array))]
        upper_met = metric_array[int(0.975*len(metric_array))]
        # print(metric, mean_met, lower_met, upper_met)
        df_sum.loc[str(metric)] = [mean_met, lower_met, upper_met]
        
    df_sum.to_excel(directories.outfile_dir + dfname + '_mean_ci_report.xlsx')
    
def test_scores(Y_true, Y_pred):
    accuracy, sens, spec, ppv, npv, auc = conmatscores(Y_true, Y_pred)
    print ("accuracy: ", round(accuracy*100,2))
    print ("sensitivity: ", round(sens*100,2))
    print ("specificity: ", round(spec*100,2))
    print ("positive predictive value: ", round(ppv*100,2))
    print ("negative predictive value: ", round(npv*100,2))
    print ("Area-under-the-curve:", round(auc,3))

def write_report(metric_type: str, Y_true, Y_pred):
    if metric_type == 'train':
        logger.info("--TRAINING METRICS--")
    elif metric_type == 'test':
        logger.info("--TESTING METRICS--")
    accuracy, sens, spec, ppv, npv, auc = conmatscores(Y_true, Y_pred)
    auc_score = ("AUC: " + str(round(auc,3)))
    sens_score = ("sensitivity: " + str(round(sens*100,2)))
    spec_score = ("specificity: " + str(round(spec*100,2)))
    ppv_score = ("positive predictive value: " + str(round(ppv*100,2)))
    npv_score = ("negative predictive value: " + str(round(npv*100,2)))
    acc_score = ("accuracy: " + str(round(accuracy*100,2)))
    logger.info(auc_score)
    logger.info(sens_score)
    logger.info(spec_score)
    logger.info(ppv_score)
    logger.info(npv_score)
    logger.info(acc_score)

#================== calculate metrics ==================
logger.info("calulating metrics")
#AD vs DLB
logger.info("AD vs DLB")
#advdlb=pickle.load(open(directories.infile_dir + 'ad_v_dlb.pkl', 'rb'))
#train
df_tr_trim_addlb = df_tr_trim.loc[(df_tr_trim.GroupID!=3) & (df_tr_trim.GroupID!=4)]
df_tr_trim_addlb_redef = df_tr_trim_addlb.replace(dict_list[1])
Y_true_tr_addlb = np.array(df_tr_trim_addlb_redef.GroupID)
Y_pred_tr_addlb = np.array(df_tr_trim_addlb_redef["dmri_ad_v_dlb (AD Probability)"])
bootstrapper(Y_true_tr_addlb, Y_pred_tr_addlb, savename="ad_v_dlb_train")
report_mean_ci(dfname="ad_v_dlb_train")
test_scores(Y_true_tr_addlb, Y_pred_tr_addlb)
write_report('train', Y_true_tr_addlb, Y_pred_tr_addlb)

#test
df_te_trim_addlb = df_te_trim.loc[(df_te_trim.GroupID!=3) & (df_te_trim.GroupID!=4)]
df_te_trim_addlb_redef = df_te_trim_addlb.replace(dict_list[1])
Y_true_te_addlb = np.array(df_te_trim_addlb_redef.GroupID)
Y_pred_te_addlb = np.array(df_te_trim_addlb_redef["dmri_ad_v_dlb (AD Probability)"])
bootstrapper(Y_true_te_addlb, Y_pred_te_addlb, savename="ad_v_dlb_test")
report_mean_ci(dfname="ad_v_dlb_test")
test_scores(Y_true_te_addlb, Y_pred_te_addlb)
write_report('test', Y_true_te_addlb, Y_pred_te_addlb)




# exit sys
sys.exit()