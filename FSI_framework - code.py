##########################################################################################################################
## Title: Heterogeneous network propagation with forward similarity integration to enhance drug–target association prediction 
## Authors: Piyanut Tangmanussukum, Thitipong Kawichai, Apichat Suratanee, and Kitiporn Plaimas

## Program Name: FSI Framework.py
## Program Description: - This program is to demonstrate how Forward Similarity Integration (FSI) framework by using smaller data sets than those original ones.
## The Forward Similarity Integration (FSI) framework is newly introduced heterogeneous network for systematically selecting drug and target similarity measures integrated into a model. 
## To enhance the heterogeneous network model by integrating multiple similarity measures of drugs and target proteins for predicting DTIs, we proposed the Forward Similarity Integration (FSI) algorithm, which systematically selects optimal similarity integration using the forward selection technique 


## Input: 1) Drug-drug similarity matrices based on chemical structures,  drug-drug interactions, drug-disease associations, and drug side effects 
##        2) Target-target similarity matrices based on protein sequence, gene ontology (GO) annotation, protein-protein interaction network, and protein pathways
##        3) Drug-target interaction matrix
##        4) Similarity network fusion (SNF) function 

## Output: 1) An optimal subset of drug-drug similarity measures
##         2) An optimal subset of target-target similarity measures
##         3) Performance measure results of all models
## 
#########################################################################################################################
#The code is written in Python (version > 3.10). 
#The packages that are required to run this code include...

#pip install numpy (in cmd)
import numpy as np
from numpy.lib.function_base import average
from numpy.lib.shape_base import kron
#pip install pandas (in cmd)
import pandas as pd
from numpy import linalg as la
import math
import csv

#pip install scikit-learn (in cmd)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score, make_scorer
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix
#pip install matplotlib (in cmd)
import matplotlib.pyplot as plt
from statistics import mean
import statistics
import numpy as geek  
import snf


## define function of heterogeneous network propagation 
def heterogeneous_network_propagation(drug_target_matrix, Wrr, Wtt, alpha, dt_test_index):
    
    # Wrr is a drug-drug similarity matrix.
    # Wtt is a target-target similarity matrix.
    # Wrt is a drug-target interaction matrix.
    # alpha or decay factor is a parameter in heterogeneous network propagation model.   
    # dt_test_index is a list of dtug-target interaction.

    # Hide known drug-target interactions in test set
    train_drug_target_matrix = np.copy(drug_target_matrix)
    for kk in range(dt_test_index.shape[0]):
        train_drug_target_matrix[dt_test_index[kk,0],dt_test_index[kk,1]] = 0

    #set parameter for stoping iterations exceeded
    max_k = 1000                 #number of maximum iterations
    max_norm = 0.001             #maximum norm
    
    # create Wrt matrix for collecting all new DTI weight matrices in every iteration 
    Wrt = np.zeros((max_k, train_drug_target_matrix.shape[1], train_drug_target_matrix.shape[0]), dtype=float)
    # set initial Wrt
    Wrt[0, :, :] = train_drug_target_matrix.T

    # normalized drug-drug similarity matrix and target-target similarity matrix
    denominator_Wrr = np.sqrt(np.sum(Wrr, axis = 0).reshape((1, Wrr.shape[1]))*np.sum(Wrr, axis = 1).reshape((Wrr.shape[0], 1)))
    denominator_Wtt = np.sqrt(np.sum(Wtt, axis = 0).reshape((1, Wtt.shape[1]))*np.sum(Wtt, axis = 1).reshape((Wtt.shape[0], 1)))

    # re-check / avoid zero values in matrices  
    denominator_Wrr[denominator_Wrr == 0] = 0.0000000001
    denominator_Wtt[denominator_Wtt == 0] = 0.0000000001

    normalized_Wrr = Wrr/denominator_Wrr
    normalized_Wtt = Wtt/denominator_Wtt

    k = 0    # set k value for initial iteration 
    while True:
        #print("k = ", k) 
        
        # calculate new values of DTIs  
        Wrt_k = np.dot(normalized_Wrr, np.dot(Wrt[k, :, :], normalized_Wtt))
        
        # normalized new values of DTI matrix 
        denominator_Wrt_k = np.sqrt(np.sum(Wrt_k, axis = 0).reshape((1, Wrt_k.shape[1]))*np.sum(Wrt_k, axis = 1).reshape((Wrt_k.shape[0], 1)))
        denominator_Wrt_k[denominator_Wrt_k == 0] = 0.0000000001

        normalized_Wrt_k = Wrt_k/denominator_Wrt_k
        
        # update new values of DTIs 
        Wrt[k+1, :, :] = alpha*normalized_Wrt_k + (1 - alpha)*Wrt[0, :, :]
    

        # check the number of maximum iterations and the maximum norm
        if la.norm(Wrt[k+1, :, :] - Wrt[k, :, :]) <= max_norm or k >= (max_k - 1) :
            break
        else:
            k = k + 1
    
    # prepare the result for export
    final_drug_target_matrix = np.copy(Wrt[k+1, :, :].T) 
    return (final_drug_target_matrix)


## define function of retrieving results of evaluation matrices  
def result_hnp(index_rrtt, kf10, drug_target_matrix, drug_similarity_matrix, target_similarity_matrix , alpha, dt_list_with_class):
    
    # create the required empty set 
    results = []             # use to collect results of performance measurements in each cross-validation round
    results_avg = []         # use to collect the average results of performance measurements
    results_var = []         # use to collect the variance results of performance measurements

    for ii in range(drug_similarity_matrix.shape[0]):
        for jj in range(target_similarity_matrix.shape[0]):
            print('ii : ',ii, ' - jj :',jj)
            n_splits_k = 0;                    # set initial counter of the splits_k
            results_round_ii = []              
            for train_index, test_index in kf10.split(dt_list_with_class):  # apply ten-folds cross-validation technique
                dt_train_index = dt_list_with_class[train_index]            # training set of DTIs
                dt_test_index = dt_list_with_class[test_index]              # testing set of DTIs

                # generate heterogeneous_network_propagation function for obtaining predicted DTIs weight matrix 
                final_drug_target_matrix = heterogeneous_network_propagation(drug_target_matrix, drug_similarity_matrix[ii,:,:], target_similarity_matrix[jj,:,:], alpha, dt_test_index)
                
                # Initialize the lists of predicted DTIs
                predicted_dt_list = np.zeros(test_index.shape, dtype=float)
                for kk in range(dt_test_index.shape[0]):
                    # create the list of predicted DTIs
                    predicted_dt_list[kk] = final_drug_target_matrix[dt_test_index[kk,0],dt_test_index[kk,1]]
                label_dt_list = dt_list_with_class[test_index]        # create the list of DTI label
                #print("predicted_dt_list :", predicted_dt_list)
                
                AUPR = average_precision_score(label_dt_list[:,2], predicted_dt_list)      # calculate area under a precision-recall curve (AUPR)
                AUC  = roc_auc_score(label_dt_list[:,2], predicted_dt_list)                # calculate area under a receiver operating characteristic curve (AUC)
                
                ## compute a binary class for each unlabeled drug-target pairs using a threshold score awarding the maximum F1
                ## retrieve precision, recall for each threshold score of the precision-recall curve
                precision, recall, threshold_pr = precision_recall_curve(label_dt_list[:,2], predicted_dt_list)
                f1 = 2 * (precision*recall) / (precision+recall)   # calculate f1-measure  
                max_index = np.argwhere(f1==max(f1))[0]                        
                threshold = threshold_pr[max_index]                # obtain the threshold that maximizes f1                   

                # define new class for predicted DTIs  
                y_pre = np.copy(predicted_dt_list)
                y_pre[y_pre >= threshold] = 1                       
                y_pre[y_pre < threshold] = 0                        
                y_pre = y_pre.astype(int)
                
                PRE = precision_score(label_dt_list[:,2], y_pre)    # calculate precision (PRE)
                REC = recall_score(label_dt_list[:,2], y_pre)       # calculate recall (REC)
                F1 = f1_score(label_dt_list[:,2], y_pre)            # calculate F1-measure
                ACC = accuracy_score(label_dt_list[:,2], y_pre)     # calculate accuracy (ACC)
                MCC = matthews_corrcoef(label_dt_list[:,2], y_pre)  # calculate Matthews Correlation Coefficient (MCC)
                
                # collect the results of performance measures
                results.append([str(index_rrtt[0])[1:-1], index_rrtt[1][ii], str(index_rrtt[2])[1:-1], index_rrtt[3][jj], AUPR, AUC, PRE, REC, F1, ACC, MCC])
                results_round_ii.append([AUPR, AUC, PRE, REC, F1, ACC, MCC])
                #AUC_sum = AUC_sum + AUC
                #print('results : ', results)
                n_splits_k = n_splits_k + 1            # update the counter of the splits_k
                print('n_splits_k : ', n_splits_k)
            #AUC_average[ii,jj] =  AUC_sum/n_splits_k
            temp_mean = np.array(results_round_ii).mean(axis=0).tolist()
            temp_var = np.array(results_round_ii).var(axis=0).tolist()
            results_avg.append([str(index_rrtt[0])[1:-1], index_rrtt[1][ii], str(index_rrtt[2])[1:-1], index_rrtt[3][jj], temp_mean[0],temp_mean[1],temp_mean[2],temp_mean[3],temp_mean[4],temp_mean[5],temp_mean[6]]) 
            results_var.append([str(index_rrtt[0])[1:-1], index_rrtt[1][ii], str(index_rrtt[2])[1:-1], index_rrtt[3][jj], temp_var[0],temp_var[1],temp_var[2],temp_var[3],temp_var[4],temp_var[5],temp_var[6]])
    #AUC_average_df = pd.DataFrame(AUC_average)
    results_df = pd.DataFrame(results, columns=cols)                     # result of all performance measurements for predicting new DTIs
    results_avg_df = pd.DataFrame(results_avg, columns=cols_avg)         # result of the performance measurement average for predicting new DTIs
    results_var_df = pd.DataFrame(results_var, columns=cols_var)         # result of the performance measurement variance for predicting new DTIs
    #print('results_df : ',results_df)    
    #results_all[ii,jj,:,:] = results_df
    return(results_df, results_avg_df, results_var_df)     


## define average function for integrating multiple similarity measures
def avg(drug_similarity_matrix, drug_used, target_similarity_matrix, target_used):
    ii = drug_used               # initialize index of drug-drug similarity measures that used to integrate
    jj = target_used             # initialize index of target-target similarity measures that used to integrate

    remaining_index_d = np.arange(drug_similarity_matrix.shape[0])      # initialize the remaining index of drug-drug similarity measures that used to integrate
    
    # remove drug-drug similarity measure that the drug data duplicated with the used drug-drug similarity measure
    for i in ii : 
        print('i : ',i)
        if i == [0] :   # index of drug-drug similarity measure based on chemical structures
            remaining_index_d = np.setdiff1d(remaining_index_d, 0)
        if i == [1] or i == [2] :    # index of drug-drug similarity measures based on drug-disease associations
            remaining_index_d = np.setdiff1d(remaining_index_d, 1)
            remaining_index_d = np.setdiff1d(remaining_index_d, 2)
        if i == [3] or i == [4] :    # index of drug-drug similarity measures based on drug-drug interactions
            remaining_index_d = np.setdiff1d(remaining_index_d, 3)
            remaining_index_d = np.setdiff1d(remaining_index_d, 4)
        if i == [5] or i == [6] :    # index of drug-drug similarity measures based on drug side effects
            remaining_index_d = np.setdiff1d(remaining_index_d, 5)
            remaining_index_d = np.setdiff1d(remaining_index_d, 6)


    # remove target-target similarity measure that the target data duplicated with the used target-target similarity measure
    remaining_index_t = np.arange(target_similarity_matrix.shape[0])
    for j in jj : 
        print('j : ',j)
        if j == [0] or j == [5] or j == [6] :   # index of target-target similarity measure based on PPI network
            remaining_index_t = np.setdiff1d(remaining_index_t, 0)  
            remaining_index_t = np.setdiff1d(remaining_index_t, 5)
            remaining_index_t = np.setdiff1d(remaining_index_t, 6)
        if j == [1] or j == [2] :   # index of target-target similarity measure based on gene ontology (GO) annotation
            remaining_index_t = np.setdiff1d(remaining_index_t, 1)
            remaining_index_t = np.setdiff1d(remaining_index_t, 2)
        if j == [3] or j == [4] :   # index of target-target similarity measure based on protein-protein interaction network
            remaining_index_t = np.setdiff1d(remaining_index_t, 3)
            remaining_index_t = np.setdiff1d(remaining_index_t, 4)
        if j == [7] or j == [8] :   # index of target-target similarity measure based on protein pathways
            remaining_index_t = np.setdiff1d(remaining_index_t, 7)
            remaining_index_t = np.setdiff1d(remaining_index_t, 8)
            
    base_sim_matrix_d = drug_similarity_matrix[ii[0],:,:]           # optimal subset of drug-drug similarity measures
    base_sim_matrix_t = target_similarity_matrix[jj[0],:,:]         # optimal subset of target-target similarity measures
    
    count_none_ii = ii.count(['None'])
    count_none_jj = jj.count(['None'])

    for ll in range(count_none_ii) :
        ii.remove(['None'])
    for ll in range(count_none_jj) :
         jj.remove(['None'])


    # integrate similarity measures where are in optimal subset 
    for ll in ii:
        base_sim_matrix_d = (base_sim_matrix_d + drug_similarity_matrix[ll,:,:])/2 
    for ll in jj:
        base_sim_matrix_t = (base_sim_matrix_t + target_similarity_matrix[ll,:,:])/2


    # integrate similarity measures where are not in optimal subset  
    rr_mat = np.zeros([remaining_index_d.shape[0], drug_similarity_matrix.shape[1], drug_similarity_matrix.shape[2]], dtype=float)
    count = 0
    for ll in remaining_index_d:
        print('ll = ', ll)
        rr_mat[count,:,:] = (base_sim_matrix_d +  drug_similarity_matrix[ll,:,:])/2
        count = count + 1

    tt_mat = np.zeros([remaining_index_t.shape[0], target_similarity_matrix.shape[1], target_similarity_matrix.shape[2]], dtype=float)
    count = 0
    for ll in remaining_index_t:
        print('ll = ', ll)
        tt_mat[count,:,:] = (base_sim_matrix_t +  target_similarity_matrix[ll,:,:])/2
        count = count + 1
    
    return(remaining_index_d, remaining_index_t, base_sim_matrix_d, base_sim_matrix_t,rr_mat,tt_mat)


## define minimum function for integrating multiple similarity measures
def mini(drug_similarity_matrix, drug_used, target_similarity_matrix, target_used):
    ii = drug_used               # initialize index of drug-drug similarity measures that used to integrate
    jj = target_used             # initialize index of target-target similarity measures that used to integrate

    remaining_index_d = np.arange(drug_similarity_matrix.shape[0]) # initialize the remaining index of drug-drug similarity measures that used to integrate
    
    # remove drug-drug similarity measure that the drug data duplicated with the used drug-drug similarity measure
    for i in ii : 
        print('i : ',i)
        if i == [0] :   # index of drug-drug similarity measure based on chemical structures
            remaining_index_d = np.setdiff1d(remaining_index_d, 0)
        if i == [1] or i == [2] :   # index of drug-drug similarity measures based on drug-disease associations 
            remaining_index_d = np.setdiff1d(remaining_index_d, 1)
            remaining_index_d = np.setdiff1d(remaining_index_d, 2)
        if i == [3] or i == [4] :   # index of drug-drug similarity measures based on drug-drug interactions
            remaining_index_d = np.setdiff1d(remaining_index_d, 3)
            remaining_index_d = np.setdiff1d(remaining_index_d, 4)
        if i == [5] or i == [6] :   # index of drug-drug similarity measures based on drug side effects
            remaining_index_d = np.setdiff1d(remaining_index_d, 5)
            remaining_index_d = np.setdiff1d(remaining_index_d, 6)

    remaining_index_t = np.arange(target_similarity_matrix.shape[0])    # remove target-target similarity measure that the target data duplicated with the used target-target similarity measure
    for j in jj : 
        print('j : ',j)
        if j == [0] or j == [5] or j == [6] :    # index of target-target similarity measure based on PPI network
            remaining_index_t = np.setdiff1d(remaining_index_t, 0)
            remaining_index_t = np.setdiff1d(remaining_index_t, 5)
            remaining_index_t = np.setdiff1d(remaining_index_t, 6)
        if j == [1] or j == [2] :    # index of target-target similarity measure based on gene ontology (GO) annotation
            remaining_index_t = np.setdiff1d(remaining_index_t, 1)
            remaining_index_t = np.setdiff1d(remaining_index_t, 2)
        if j == [3] or j == [4] :    # index of target-target similarity measure based on protein-protein interaction network
            remaining_index_t = np.setdiff1d(remaining_index_t, 3)
            remaining_index_t = np.setdiff1d(remaining_index_t, 4)
        if j == [7] or j == [8] :    # index of target-target similarity measure based on protein pathways
            remaining_index_t = np.setdiff1d(remaining_index_t, 7)
            remaining_index_t = np.setdiff1d(remaining_index_t, 8)
            
    base_sim_matrix_d = drug_similarity_matrix[ii[0],:,:]       # optimal subset of drug-drug similarity measures
    base_sim_matrix_t = target_similarity_matrix[jj[0],:,:]     # optimal subset of target-target similarity measures
    
    count_none_ii = ii.count(['None'])
    count_none_jj = jj.count(['None'])

    for ll in range(count_none_ii) :
        ii.remove(['None'])
    for ll in range(count_none_jj) :
         jj.remove(['None'])


    # integrate similarity measures where are in optimal subset 
    for ll in ii:
        base_sim_matrix_d = geek.minimum(base_sim_matrix_d, drug_similarity_matrix[ll,:,:])
    for ll in jj:
        base_sim_matrix_t = geek.minimum(base_sim_matrix_t, target_similarity_matrix[ll,:,:])


    # integrate similarity measures where are not in optimal subset
    rr_mat = np.zeros([remaining_index_d.shape[0], drug_similarity_matrix.shape[1], drug_similarity_matrix.shape[2]], dtype=float)
    count = 0
    for ll in remaining_index_d:
        print('ll = ', ll)
        rr_mat[count,:,:] = geek.minimum(base_sim_matrix_d, drug_similarity_matrix[ll,:,:])
        count = count + 1

    tt_mat = np.zeros([remaining_index_t.shape[0], target_similarity_matrix.shape[1], target_similarity_matrix.shape[2]], dtype=float)
    count = 0
    for ll in remaining_index_t:
        print('ll = ', ll)
        tt_mat[count,:,:] = geek.minimum(base_sim_matrix_t, target_similarity_matrix[ll,:,:])
        count = count + 1
    
    return(remaining_index_d, remaining_index_t, base_sim_matrix_d, base_sim_matrix_t, rr_mat, tt_mat)


## define maximum function for integrating multiple similarity measures
def maxi(drug_similarity_matrix, drug_used, target_similarity_matrix, target_used):
    ii = drug_used               # initialize index of drug-drug similarity measures that used to integrate
    jj = target_used             # initialize index of target-target similarity measures that used to integrate

    remaining_index_d = np.arange(drug_similarity_matrix.shape[0])    # initialize the remaining index of drug-drug similarity measures that used to integrate
    
    # remove drug-drug similarity measure that the drug data duplicated with the used drug-drug similarity measure
    for i in ii : 
        print('i : ',i)
        if i == [0] :     # index of drug-drug similarity measure based on chemical structures
            remaining_index_d = np.setdiff1d(remaining_index_d, 0)
        if i == [1] or i == [2] :     # index of drug-drug similarity measures based on drug-disease associations
            remaining_index_d = np.setdiff1d(remaining_index_d, 1)
            remaining_index_d = np.setdiff1d(remaining_index_d, 2)
        if i == [3] or i == [4] :     # index of drug-drug similarity measures based on drug-drug interactions
            remaining_index_d = np.setdiff1d(remaining_index_d, 3)
            remaining_index_d = np.setdiff1d(remaining_index_d, 4)
        if i == [5] or i == [6] :     # index of drug-drug similarity measures based on drug side effects
            remaining_index_d = np.setdiff1d(remaining_index_d, 5)
            remaining_index_d = np.setdiff1d(remaining_index_d, 6)

    # remove target-target similarity measure that the target data duplicated with the used target-target similarity measure
    remaining_index_t = np.arange(target_similarity_matrix.shape[0])
    for j in jj : 
        print('j : ',j)
        if j == [0] or j == [5] or j == [6] :    # index of target-target similarity measure based on PPI network
            remaining_index_t = np.setdiff1d(remaining_index_t, 0)
            remaining_index_t = np.setdiff1d(remaining_index_t, 5)
            remaining_index_t = np.setdiff1d(remaining_index_t, 6)
        if j == [1] or j == [2] :    # index of target-target similarity measure based on gene ontology (GO) annotation
            remaining_index_t = np.setdiff1d(remaining_index_t, 1)
            remaining_index_t = np.setdiff1d(remaining_index_t, 2)
        if j == [3] or j == [4] :    # index of target-target similarity measure based on protein-protein interaction network
            remaining_index_t = np.setdiff1d(remaining_index_t, 3)
            remaining_index_t = np.setdiff1d(remaining_index_t, 4)
        if j == [7] or j == [8] :    # index of target-target similarity measure based on protein pathways
            remaining_index_t = np.setdiff1d(remaining_index_t, 7)
            remaining_index_t = np.setdiff1d(remaining_index_t, 8)
            
    base_sim_matrix_d = drug_similarity_matrix[ii[0],:,:]      # optimal subset of drug-drug similarity measures
    base_sim_matrix_t = target_similarity_matrix[jj[0],:,:]    # optimal subset of target-target similarity measures
    
    count_none_ii = ii.count(['None'])
    count_none_jj = jj.count(['None'])

    for ll in range(count_none_ii) :
        ii.remove(['None'])
    for ll in range(count_none_jj) :
         jj.remove(['None'])

    # integrate similarity measures where are in optimal subset 
    for ll in ii:
        base_sim_matrix_d = geek.maximum(base_sim_matrix_d, drug_similarity_matrix[ll,:,:])
    for ll in jj:
        base_sim_matrix_t = geek.maximum(base_sim_matrix_t, target_similarity_matrix[ll,:,:])

    # integrate similarity measures where are not in optimal subset  
    rr_mat = np.zeros([remaining_index_d.shape[0], drug_similarity_matrix.shape[1], drug_similarity_matrix.shape[2]], dtype=float)
    count = 0
    for ll in remaining_index_d:
        print('ll = ', ll)
        rr_mat[count,:,:] = geek.maximum(base_sim_matrix_d, drug_similarity_matrix[ll,:,:])
        count = count + 1

    tt_mat = np.zeros([remaining_index_t.shape[0], target_similarity_matrix.shape[1], target_similarity_matrix.shape[2]], dtype=float)
    count = 0
    for ll in remaining_index_t:
        print('ll = ', ll)
        tt_mat[count,:,:] = geek.maximum(base_sim_matrix_t, target_similarity_matrix[ll,:,:])
        count = count + 1
    
    return(remaining_index_d, remaining_index_t, base_sim_matrix_d, base_sim_matrix_t,rr_mat,tt_mat)


## define SNF function for integrating multiple similarity measures
def my_snf(drug_used, target_used):

    drug_sim_mat = [rr0,rr1,rr2,rr3,rr4,rr5,rr6]                    # initialize index of drug-drug similarity measures that used to integrate
    target_sim_mat = [tt0,tt1,tt2,tt3,tt4,tt5,tt6,tt7,tt8]          # initialize index of target-target similarity measures that used to integrate
    
    ii = drug_used      
    jj = target_used

    remaining_index_d = np.arange(len(drug_sim_mat))     # initialize the remaining index of drug-drug similarity measures that used to integrate
    
    # remove drug-drug similarity measure that the drug data duplicated with the used drug-drug similarity measure
    for i in ii : 
        print('i : ',i)
        if i == [0] :    # index of drug-drug similarity measure based on chemical structures
            remaining_index_d = np.setdiff1d(remaining_index_d, 0)
        if i == [1] or i == [2] :    # index of drug-drug similarity measures based on drug-disease associations
            remaining_index_d = np.setdiff1d(remaining_index_d, 1)
            remaining_index_d = np.setdiff1d(remaining_index_d, 2)
        if i == [3] or i == [4] :    # index of drug-drug similarity measures based on drug-drug interactions
            remaining_index_d = np.setdiff1d(remaining_index_d, 3)
            remaining_index_d = np.setdiff1d(remaining_index_d, 4)
        if i == [5] or i == [6] :    # index of drug-drug similarity measures based on drug side effects
            remaining_index_d = np.setdiff1d(remaining_index_d, 5)
            remaining_index_d = np.setdiff1d(remaining_index_d, 6)


    # remove target-target similarity measure that the target data duplicated with the used target-target similarity measure
    remaining_index_t = np.arange(len(target_sim_mat))
    for j in jj : 
        print('j : ',j)
        if j == [0] or j == [5] or j == [6] :    # index of target-target similarity measure based on PPI network
            remaining_index_t = np.setdiff1d(remaining_index_t, 0)
            remaining_index_t = np.setdiff1d(remaining_index_t, 5)
            remaining_index_t = np.setdiff1d(remaining_index_t, 6)
        if j == [1] or j == [2] :    # index of target-target similarity measure based on gene ontology (GO) annotation
            remaining_index_t = np.setdiff1d(remaining_index_t, 1)
            remaining_index_t = np.setdiff1d(remaining_index_t, 2)
        if j == [3] or j == [4] :    # index of target-target similarity measure based on protein-protein interaction network
            remaining_index_t = np.setdiff1d(remaining_index_t, 3)
            remaining_index_t = np.setdiff1d(remaining_index_t, 4)
        if j == [7] or j == [8] :    # index of target-target similarity measure based on protein pathways
            remaining_index_t = np.setdiff1d(remaining_index_t, 7)
            remaining_index_t = np.setdiff1d(remaining_index_t, 8)
            
    count_none_ii = ii.count(['None'])
    count_none_jj = jj.count(['None'])

    for ll in range(count_none_ii) :
        ii.remove(['None'])
    for ll in range(count_none_jj) :
         jj.remove(['None'])

    ### main similarity area ###
    Wall_d = []
    for ll in range(len(ii)) :
        print(ii[ll][0])
        Wall_d.append(drug_sim_mat[ii[ll][0]])

    Wall_t = []
    for ll in range(len(jj)) :
        print(jj[ll][0])
        Wall_t.append(target_sim_mat[jj[ll][0]])

    if len(ii) == 1 :
        base_sim_matrix_d = drug_sim_mat[ii[0][0]]
    else : 
        base_sim_matrix_d = snf.snf(Wall_d, K=20)

    if len(jj) == 1 :
        base_sim_matrix_t = target_sim_mat[jj[0][0]]
    else : 
        base_sim_matrix_t = snf.snf(Wall_t, K=20)
        
    test = snf.snf(rr2,rr3,rr0, K=20)
    base_sim_matrix_d == test

    ### remaining similarity area ###
    Wall_remaining_d = []
    for ll in range(len(remaining_index_d)) :
        print(remaining_index_d[ll])
        Wall_remaining_d.append(drug_sim_mat[remaining_index_d[ll]])

    Wall_remaining_t = []
    for ll in range(len(remaining_index_t)) :
        print(remaining_index_t[ll])
        Wall_remaining_t.append(target_sim_mat[remaining_index_t[ll]])

    rr_mat = np.zeros([remaining_index_d.shape[0], drug_sim_mat[0].shape[0], drug_sim_mat[0].shape[1]], dtype=float)
    for ll in range(len(remaining_index_d)) :
        print('ll = ', ll)
        rr_mat[ll,:,:] = snf.snf(Wall_d, Wall_remaining_d[ll], K=20)

    tt_mat = np.zeros([remaining_index_t.shape[0], target_sim_mat[0].shape[0], target_sim_mat[0].shape[1]], dtype=float)
    for ll in range(len(remaining_index_t)) :
        print('ll = ', ll)
        tt_mat[ll,:,:] = snf.snf(Wall_t, Wall_remaining_t[ll], K=20)

    base_sim_matrix_3d = np.zeros([1, drug_sim_mat[0].shape[0], drug_sim_mat[0].shape[1]], dtype=float)
    base_sim_matrix_3t = np.zeros([1, target_sim_mat[0].shape[0], target_sim_mat[0].shape[1]], dtype=float)
    base_sim_matrix_3d[0,:,:] = base_sim_matrix_d
    base_sim_matrix_3t[0,:,:] = base_sim_matrix_t

    return(remaining_index_d, remaining_index_t, base_sim_matrix_3d, base_sim_matrix_3t, rr_mat, tt_mat)


## define FSI function for selecting optimal drug similarity measures and target similarity measures 
def forward_similarity_integration(alpha, type_integrate_sim, type_max_value, results_df, results_avg_df, results_var_df) :
    results_ff = results_df
    results_avg_ff = results_avg_df
    results_var_ff = results_var_df

    # option of similarity integration ####
    if type_max_value == 'AUC' :
        max_value = results_avg_df.iloc[results_avg_df[['AUC_avg']].idxmax()]['AUC_avg'].tolist()
    if type_max_value == 'AUPR' :
        max_value = results_avg_df.iloc[results_avg_df[['AUPR_avg']].idxmax()]['AUPR_avg'].tolist()
    if type_max_value == 'F1' :
        max_value = results_avg_df.iloc[results_avg_df[['F1_avg']].idxmax()]['F1_avg'].tolist()

    max_value_before = [0]       # initialize maximum value of the performance measure
    drug_used = []               # index of drug-drug similarity measures in optimal set
    target_used = []             # index of target-target similarity measures in optimal set
    
    # select the performance measure function
    if type_max_value == 'AUC' :
        drug_used.append(results_avg_df.iloc[results_avg_df[['AUC_avg']].idxmax()]['drug sim added'].tolist())
        target_used.append(results_avg_df.iloc[results_avg_df[['AUC_avg']].idxmax()]['target sim added'].tolist())
                
    if type_max_value == 'AUPR' :
        drug_used.append(results_avg_df.iloc[results_avg_df[['AUPR_avg']].idxmax()]['drug sim added'].tolist())
        target_used.append(results_avg_df.iloc[results_avg_df[['AUPR_avg']].idxmax()]['target sim added'].tolist())
    
    if type_max_value == 'F1' :
        drug_used.append(results_avg_df.iloc[results_avg_df[['F1_avg']].idxmax()]['drug sim added'].tolist())
        target_used.append(results_avg_df.iloc[results_avg_df[['F1_avg']].idxmax()]['target sim added'].tolist())
 
    len_remain_index_d = 1
    len_remain_index_t = 1

    # systematically select optimal drug similarity measures and target similarity measures 
    while max_value[0] > max_value_before[0] and len_remain_index_d != 0 and len_remain_index_t != 0 :
        max_value_before = max_value
        
        if type_integrate_sim == 'AVG':  #Average function
            remaining_index_d, remaining_index_t, base_sim_matrix_d, base_sim_matrix_t, rr_mat, tt_mat = avg(drug_similarity_matrix, drug_used, target_similarity_matrix, target_used)
    
        if type_integrate_sim == 'MAX':  #Maximum function
            remaining_index_d, remaining_index_t, base_sim_matrix_d, base_sim_matrix_t, rr_mat, tt_mat = maxi(drug_similarity_matrix, drug_used, target_similarity_matrix, target_used)
     
        if type_integrate_sim == 'MIN':  #Minimum function
            remaining_index_d, remaining_index_t, base_sim_matrix_d, base_sim_matrix_t, rr_mat, tt_mat = mini(drug_similarity_matrix, drug_used, target_similarity_matrix, target_used)

        if type_integrate_sim == 'SNF':  #SNF function
            remaining_index_d, remaining_index_t, base_sim_matrix_d, base_sim_matrix_t, rr_mat, tt_mat = my_snf(drug_used, target_used)
        
        len_remain_index_d = len(remaining_index_d)         #  
        len_remain_index_t = len(remaining_index_t)         #

        index_rrtt =  drug_used, ['None'], target_used, remaining_index_t
        results_df_1, results_avg_df_1, results_var_df_1 = result_hnp(index_rrtt, kf10, drug_target_matrix, base_sim_matrix_d, tt_mat, alpha, dt_list_with_class)

        index_rrtt =  drug_used, remaining_index_d, target_used, ['None']
        results_df_2, results_avg_df_2, results_var_df_2 = result_hnp(index_rrtt, kf10, drug_target_matrix, rr_mat, base_sim_matrix_t, alpha, dt_list_with_class)
        
        index_rrtt =  drug_used, remaining_index_d, target_used, remaining_index_t
        results_df_3, results_avg_df_3, results_var_df_3 = result_hnp(index_rrtt, kf10, drug_target_matrix, rr_mat, tt_mat, alpha, dt_list_with_class)
        
        avg_temp = pd.concat([pd.DataFrame(results_avg_df_1.append(results_avg_df_2).append(results_avg_df_3))]).reset_index(drop=True)

        if type_max_value == 'AUC' :
            drug_used.append(avg_temp.iloc[avg_temp[['AUC_avg']].idxmax()]['drug sim added'].tolist())
            target_used.append(avg_temp.iloc[avg_temp[['AUC_avg']].idxmax()]['target sim added'].tolist())
            max_value = avg_temp.iloc[avg_temp[['AUC_avg']].idxmax()]['AUC_avg'].tolist()

        if type_max_value == 'AUPR' :
            drug_used.append(avg_temp.iloc[avg_temp[['AUPR_avg']].idxmax()]['drug sim added'].tolist())
            target_used.append(avg_temp.iloc[avg_temp[['AUPR_avg']].idxmax()]['target sim added'].tolist())
            max_value = avg_temp.iloc[avg_temp[['AUPR_avg']].idxmax()]['AUPR_avg'].tolist()
        
        if type_max_value == 'F1' :
            drug_used.append(avg_temp.iloc[avg_temp[['F1_avg']].idxmax()]['drug sim added'].tolist())
            target_used.append(avg_temp.iloc[avg_temp[['F1_avg']].idxmax()]['target sim added'].tolist())
            max_value = avg_temp.iloc[avg_temp[['F1_avg']].idxmax()]['F1_avg'].tolist()

        results_ff = results_ff.append(results_df_1).append(results_df_2).append(results_df_3)
        results_avg_ff = results_avg_ff.append(results_avg_df_1).append(results_avg_df_2).append(results_avg_df_3)
        results_var_ff = results_var_ff.append(results_var_df_1).append(results_var_df_2).append(results_var_df_3)

    #if type_max_value == 'AUC' :
    #    print(results_avg_ff.iloc[results_avg_ff[['AUC_avg']].idxmax()])
    #if type_max_value == 'AUPR' :
    #    print(results_avg_ff.iloc[results_avg_ff[['AUPR_avg']].idxmax()])

    return(results_ff, results_avg_ff, results_var_ff, max_value_before)
   


if __name__ == "__main__":

    num_drugs, num_target = 862, 1517

    ### input the required files ###
    path1 = 'C:/Users/admin/OneDrive - Chulalongkorn University/work/Thesis/final similarity measure collection/'

    n_splits = 10
 
    cols = ['drug sim main', 'drug sim added', 'target sim main', 'target sim added','AUPR', 'AUC', 'PRE', 'REC', 'F1', 'ACC', 'MCC']
    cols_avg = [ 'drug sim main', 'drug sim added', 'target sim main', 'target sim added','AUPR_avg', 'AUC_avg', 'PRE_avg', 'REC_avg', 'F1_avg', 'ACC_avg', 'MCC_avg']
    cols_var = ['drug sim main', 'drug sim added', 'target sim main', 'target sim added','AUPR_var', 'AUC_var', 'PRE_var', 'REC_var', 'F1_var', 'ACC_var', 'MCC_var']
    kf10 = KFold(n_splits=n_splits, shuffle=True, random_state=18)

    rr0 = (pd.read_csv(path1 + 'drug-drug similarity by Chemical structure.csv', header = None)).to_numpy()
    rr1 = (pd.read_csv(path1 + 'drug-drug similarity by Disease - Cosine index.csv', header = None)).to_numpy()
    rr2 = (pd.read_csv(path1 + 'drug-drug similarity by Disease - Jaccard index.csv', header = None)).to_numpy()
    rr3 = (pd.read_csv(path1 + 'drug-drug similarity by Drug - Cosine index.csv', header = None)).to_numpy()
    rr4 = (pd.read_csv(path1 + 'drug-drug similarity by Drug - Jaccard index.csv', header = None)).to_numpy()
    rr5 = (pd.read_csv(path1 + 'drug-drug similarity by SideEffect - Cosine index.csv', header = None)).to_numpy()
    rr6 = (pd.read_csv(path1 + 'drug-drug similarity by SideEffect - Jaccard index.csv', header = None)).to_numpy()

    tt0 = (pd.read_csv(path1 + 'Protein-Protein similarity by PPI - inverse shortest path.csv', header = None)).to_numpy()
    tt1 = (pd.read_csv(path1 + 'Protein-Protein similarity by GOSemSim_Jiang_BMA.csv', header = None)).to_numpy()
    tt2 = (pd.read_csv(path1 + 'Protein-Protein similarity by GOSemSim_wang_BMA.csv', header = None)).to_numpy()
    tt3 = (pd.read_csv(path1 + 'Protein-Protein similarity by Matabolic - Cosine index.csv', header = None)).to_numpy()
    tt4 = (pd.read_csv(path1 + 'Protein-Protein similarity by Matabolic - Jaccard index.csv', header = None)).to_numpy()
    tt5 = (pd.read_csv(path1 + 'Protein-Protein similarity by PPI - Cosine index.csv', header = None)).to_numpy()
    tt6 = (pd.read_csv(path1 + 'Protein-Protein similarity by PPI - Jaccard index.csv', header = None)).to_numpy()
    tt7 = (pd.read_csv(path1 + 'Protein-Protein similarity by Sequence - ND_normalize_global.csv', header = None)).to_numpy()
    tt8 = (pd.read_csv(path1 + 'Protein-Protein similarity by Sequence - SW_normalize_local.csv', header = None)).to_numpy()
    
    drug_target_matrix = (pd.read_csv(path1 + 'drug-target_1517_862.csv', header = None)).to_numpy()
    dt_list_with_class = (pd.read_csv(path1 + 'list_dt_class.csv', header = None)).to_numpy()
    
    #############################################################################################################################
    #############################################################################################################################

    ### required matrices and lists for using with the function  ###
    drug_similarity_matrix = np.zeros([7, 862, 862], dtype=float)                           # list of all drug-drug similarity measures
    target_similarity_matrix = np.zeros([9, 1517, 1517], dtype=float)                       # list of all target-target similarity measures
    drug_similarity_matrix[:,:,:] = rr0, rr1, rr2, rr3, rr4, rr5, rr6                       # list of index of drug-drug similarity measures  
    target_similarity_matrix[:,:,:] = tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8           # list of index of target-target similarity measures
    index_rrtt =  ['None'], [0,1,2,3,4,5,6], ['None'], [0,1,2,3,4,5,6,7,8]      


    # Example of using FSI algorithm
                                    
    results_df, results_avg_df, results_var_df = result_hnp(index_rrtt, kf10, drug_target_matrix, drug_similarity_matrix, target_similarity_matrix , 0.4, dt_list_with_class)

    #### In here, we choose alpha = 0.1,
    #                       similarity integration method is Average function, 
    #                       performance measure is F1-measure
    results_all_ddtt_alpha1_average_f1, results_ddtt_alpha1_average_f1_avg, results_ddtt_alpha1_average_f1_var, max_ddtt_alpha1_average_f1 = forward_similarity_integration(0.1, 'AVG', 'F1', results_df, results_avg_df, results_var_df)
    results_ddtt_alpha1_average_f1_avg.iloc[results_ddtt_alpha1_average_f1_avg[['AUC_avg']].idxmax()]       # obtain average of the performance measures
    
