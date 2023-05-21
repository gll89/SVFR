import os, time, pdb
import numpy as np
from sklearn.metrics import confusion_matrix

def conf_matrx_cal(labels_gt, labels_pred):
    '''
    Input: labels_gt: ground truth;  labels_pred: predicted labels
    Spe: TN/(TN+FP)
    BAC: Balance of Specificity and Sensitivity, i.e., (Specificity+Sensitivity)/2
    ROC: Sensitivity vs 1-Specificity
    ROC Area: the area under ROC
    '''
    # inds_p_gt = labels_gt==0
    # inds_n_gt = labels_gt==1
    # valus_p_prd = labels_pred[inds_p_gt]
    # valus_n_prd = labels_pred[inds_n_gt]
    # TP, TN = np.sum(valus_p_prd==0), np.sum(valus_n_prd==1)
    # FP, FN = np.sum(labels_pred==0)-TP, np.sum(labels_pred==1)-TN
    # conf_matrx = [[TP, FN], [FP, TN]]
    conf_matrx = confusion_matrix(labels_gt, labels_pred)
    TN,FP,FN,TP = confusion_matrix(labels_gt, labels_pred).ravel()
    sen = rec = 1.00*(TP)/(TP+FN)  #recall
    spe = 1.00*(TN)/(TN+FP)  #len(inds_n_gt)
    acc = 1.00*(TP+TN)/len(labels_gt)
    pre = 1.00*(TP)/(TP+FP)  #np.sum(labels_pred==0)
    f1 = 2.00*pre*rec/(pre+rec)

    # import pdb; pdb.set_trace()
    # print('Acc: {}\tConf_matrx: {}'.format(acc, conf_matrx))
    # print('Rec\Sen: {}\tSpe: {}\tPre: {}\tF1-score: {}'.format(sen, spe, pre, f1))
    return acc, conf_matrx, sen, spe, pre, f1


# gt = np.random.randint(0,2,5)
# preds = np.random.randint(0, 2, 5)
# acc, conf_matrx, _, pre, rec, f1, spe, bac = conf_matrx_cal(gt, preds)
# print('gt: {}\npreds:{}\nacc:{}\nconf_matrx: {}\nbac:{}'.format(gt, preds, acc, conf_matrx, bac))
# print(acc, conf_matrx, bac)
