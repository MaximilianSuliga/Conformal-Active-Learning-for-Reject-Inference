#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
import scorecardpy as sc
from sklearn.model_selection import train_test_split
from nonconformist.cp import IcpClassifier
from nonconformist.nc import ClassifierNc
from nonconformist.nc import MarginErrFunc
from nonconformist.nc import InverseProbabilityErrFunc
from nonconformist.base import ClassifierAdapter
from nonconformist.nc import ClassificationErrFunc



def cost_matrix_Small(df):
    '''Cost function for the dataset "Small". Assumption of 5% interest p.a.'''
    cost_matrix = pd.DataFrame()
    cost_matrix['false_positive'] = df['LoanAmount'] * df['Loan_Amount_Term'] / 12 *0.05 #5% p.a. interest assumed
    cost_matrix['false_negative'] = df['LoanAmount']
    return cost_matrix
    
def cost_matrix_German(df):
    '''Cost function for the dataset "German". Mean monthly duration was rounded to 24.'''
    cost_matrix = pd.DataFrame()
    cost_matrix['false_positive'] = df['Duration_in_month'] /24 #mean is nearly 21, so /24 is approx 1
    cost_matrix['false_negative'] = 4
    return cost_matrix
    
def cost_matrix_Deloitte(df):
    '''Cost function for the dataset "Deloitte". Assumption of term to be documented in months.'''
    cost_matrix = pd.DataFrame()
    cost_matrix['false_positive'] = df['Loan Amount'] * df['Interest Rate'] * df['Term'] /100 /12 #term seems to be in months
    cost_matrix['false_negative'] = df['Loan Amount']
    return cost_matrix
    
def cost_matrix_Large(df):
    '''Cost function for the dataset "Large". Assumption of term to be documented in months, rate of interest in %.'''
    cost_matrix = pd.DataFrame()
    cost_matrix['false_positive'] = df['loan_amount'] * df['term'] / 12 * df['rate_of_interest'] / 100 + df['Upfront_charges']
    cost_matrix['false_negative'] = df['loan_amount']
    return cost_matrix
    
def cost_matrix_LC(df):
    '''Cost function for the dataset "LC". Since term and iterest are not available for rejected applications, 
    the most frequent term in years and the average interest rate are assumed for every application.'''
    cost_matrix = pd.DataFrame()
    cost_matrix['false_positive'] = df['Amount Requested'] * 3 * 0.13 #most frequent term in years and average interest
    cost_matrix['false_negative'] = df['Amount Requested']
    return cost_matrix
    
def cost_loss(y_true, y_pred, cost_mat):
    '''Cost Loss Function from Bahnsen (2016)'''
    y_true = y_true.values.ravel()
    y_pred = y_pred.values.ravel()
    cost_mat = cost_mat.to_numpy()
    cost = y_true * ((1 - y_pred) * cost_mat[:, 1]) #+ y_pred * cost_mat[:, 2]) fn
    cost += (1 - y_true) * (y_pred * cost_mat[:, 0]) #+ (1 - y_pred) * cost_mat[:, 3]) fp
    return np.sum(cost)

def data_prep(start_, test_, name):
    '''Data Preparation Function. Includes generating the cost efficient threshold, normalization of numerical features
    and WoE transformation of categorical features.'''
    
    # Copies of dataframe inputs
    start  = start_.copy()
    test=test_.copy()
    
    
    
    # Use appropriate cost function
    if name ==  'Small':
        cost_matrix = cost_matrix_Small(start)
        cost_matrix_test = cost_matrix_Small(test)
    elif name ==  'German':
        cost_matrix = cost_matrix_German(start)
        cost_matrix_test = cost_matrix_German(test)
    elif name ==  'Deloitte':
        cost_matrix = cost_matrix_Deloitte(start)
        cost_matrix_test = cost_matrix_Deloitte(test)
    elif name ==  'Large':
        cost_matrix = cost_matrix_Large(start)
        cost_matrix_test = cost_matrix_Large(test)
    elif name ==  'LC':
        cost_matrix = cost_matrix_LC(start)
        cost_matrix_test = cost_matrix_LC(test)
        
        

    # Calculate Bayes optimal threshold
    threshold = np.mean( cost_matrix['false_positive']) / np.mean(cost_matrix['false_negative'] + cost_matrix['false_positive'])
    
    
    # Ensure correct data type for each feature
    for col in test.columns:
        dtype = pd.api.types.infer_dtype(test[col])
        if dtype == 'categorical':
            test[col] = test[col].astype('category')
        elif dtype == 'floating':
            test[col] = test[col].astype('float')
        elif dtype == 'integer':
            test[col] = test[col].astype('float')
        
    for col in start.columns:
        dtype = pd.api.types.infer_dtype(start[col])
        if dtype == 'categorical':
            start[col] = start[col].astype('category')
        elif dtype == 'floating':
            start[col] = start[col].astype('float')
        elif dtype == 'integer':
            start[col] = start[col].astype('float')
     
    # Drop features with only 1 feature
    for col in start.columns:
        if start[col].nunique() == 1:
            start.drop(col, axis=1, inplace=True)
            test.drop(col, axis=1, inplace=True)

    # Select the categorical columns and transform to WoE
    cat_cols = start.select_dtypes(include=['object', 'category']).columns
    bins = sc.woebin(start, y="BAD", x=cat_cols.tolist(), print_step=0, check_cate_num=False)
    start_trans = sc.woebin_ply(start, bins, print_step=0)


    # Select the numeric columns and standardize them
    numeric_cols = start.drop(columns='BAD').select_dtypes(include=['float', 'int', 'uint8']).columns
    scaler = StandardScaler()
    scaler.fit(start_trans[numeric_cols])
    start_trans[numeric_cols] = scaler.transform(start_trans[numeric_cols])
    
    # Same procedur for test set
    test_trans=test
    test_trans[numeric_cols] = scaler.transform(test_trans[numeric_cols])
    test_trans = sc.woebin_ply(test_trans, bins, print_step=0)

    # If test set contained categories unavailable in the start set, replace its WoE with the most frequent one
    for feature in test_trans.columns:
        most_frequent_value = test_trans[feature].mode().values[0]
        test_trans[feature].fillna(most_frequent_value, inplace=True)
    
    return start_trans, test_trans, cost_matrix, threshold, cost_matrix_test


def no_al(start, start_trans, test, test_trans_, threshold_, cost_matrix_, cost_matrix_test_):
    '''Benchmark strategy of No Active Learning'''
    
    # Copies of used dataframes
    start_= start.copy()
    start_trans_ = start_trans.copy()
    test_ = test.copy()
    test_trans = test_trans_.copy()
    
    # Underlying model
    LR = LogisticRegression(penalty='l1', solver='saga', random_state=0, class_weight='balanced')
    LR.fit(start_trans_.drop(columns=['BAD']), start_trans_['BAD'])
    
    # Predict probabilities
    test_['BAD_pred'] = LR.predict_proba(test_trans.drop(columns=['BAD']))[:, 1]
    
    test_['BAD'] = test_['BAD'].astype(int)
    
    # Assess performance
    AUC = roc_auc_score(test_['BAD'],test_['BAD_pred'])
    
    ptest = test_[test_['BAD_pred']<0.5]
    
    if len(ptest) == 0 or len(ptest['BAD'].unique()) ==1:
        PAUC = None
    else:
        PAUC = roc_auc_score(ptest['BAD'],ptest['BAD_pred'])
        
    BS = brier_score_loss(test_['BAD'], test_['BAD_pred'])


    
    # Round probabilites to classes
    test_['BAD_pred'].loc[test_['BAD_pred'] >= threshold_] = 1
    test_['BAD_pred'].loc[test_['BAD_pred'] < threshold_] = 0
    test_['BAD_pred'] = test_['BAD_pred'].astype(int)
   
    # Assess Cost
    Cost = cost_loss(test_['BAD'], test_['BAD_pred'], cost_matrix_test_)
    metrics=[AUC, PAUC, BS, Cost, threshold_]

    # Concat accepted applications to training data
    goods_pred = test_[test_['BAD_pred']==0].drop(columns='BAD_pred')
    
    new_data = pd.concat([start, goods_pred])

    return new_data, metrics

def random_al(start, start_trans, test, test_trans_, threshold_, cost_matrix_, cost_matrix_test_):
    '''Active Learning strategy of selecting a random sample of AL instances'''
    
    # Copies of used dataframes
    start_= start.copy()
    start_trans_ = start_trans.copy()
    test_ = test.copy()
    test_trans = test_trans_.copy()
    
    # Underlying model
    LR = LogisticRegression(penalty='l1', solver='saga', random_state=0, class_weight='balanced')
    LR.fit(start_trans_.drop(columns=['BAD']), start_trans_['BAD'])
    
    # Predict probabilities
    test_['BAD_pred'] = LR.predict_proba(test_trans.drop(columns=['BAD']))[:, 1]
    
    test_['BAD'] = test_['BAD'].astype(int)
    
    # Assess performance
    AUC = roc_auc_score(test_['BAD'],test_['BAD_pred'])
    
    ptest = test_[test_['BAD_pred']<0.5]
    
    if len(ptest) == 0 or len(ptest['BAD'].unique()) ==1:
        PAUC = None
    else:
        PAUC = roc_auc_score(ptest['BAD'],ptest['BAD_pred'])
        
    BS = brier_score_loss(test_['BAD'], test_['BAD_pred'])


    

    # Round probabilites to classes
    test_['BAD_pred'].loc[test_['BAD_pred'] >= threshold_] = 1
    test_['BAD_pred'].loc[test_['BAD_pred'] < threshold_] = 0
    test_['BAD_pred'] = test_['BAD_pred'].astype(int)
    
    # Set random seed
    np.random.seed(0)

    # Randomly sample rejected applications of sample size of the BAD share of accepted applications and turn their prediction
    # from BAD to GOOD
    mask = test_['BAD_pred'] == 1
    sample_size = int(np.mean(start['BAD']) * len(test_[test_['BAD_pred'] ==0]))
    sample_size=min(sample_size, len(test_[test_['BAD_pred'] ==1]))
    al_customers = np.random.choice(test_[mask].index, size=sample_size, replace=False)

    # Change the selected 1s to 0s in the dataframe
    test_.loc[al_customers, 'BAD_pred'] = 0
    
    # Assess Cost
    Cost = cost_loss(test_['BAD'], test_['BAD_pred'], cost_matrix_test_)
    metrics=[AUC, PAUC, BS, Cost, threshold_]

    # Concat accepted applications to training data
    goods_pred = test_[test_['BAD_pred']==0].drop(columns='BAD_pred')
    new_data = pd.concat([start_, goods_pred])

    return new_data, metrics

def benchmark_ri(start, start_trans, test, test_trans_, threshold_, cost_matrix_, cost_matrix_test_):
    '''Reject Inference Benchmark of selecting the same random sample of AL instances, 
    but marking them as defaulted without handing the loan'''
    
    # Copies of used dataframes
    start_= start.copy()
    start_trans_ = start_trans.copy()
    test_ = test.copy()
    test_trans = test_trans_.copy()
    
    # Underlying model
    LR = LogisticRegression(penalty='l1', solver='saga', random_state=0, class_weight='balanced')
    LR.fit(start_trans_.drop(columns=['BAD']), start_trans_['BAD'])
    
    # Predict probabilities
    test_['BAD_pred'] = LR.predict_proba(test_trans.drop(columns=['BAD']))[:, 1]
    
    test_['BAD'] = test_['BAD'].astype(int)
    
    # Assess performance
    AUC = roc_auc_score(test_['BAD'],test_['BAD_pred'])
    
    ptest = test_[test_['BAD_pred']<0.5]
    
    if len(ptest) == 0 or len(ptest['BAD'].unique()) ==1:
        PAUC = None
    else:
        PAUC = roc_auc_score(ptest['BAD'],ptest['BAD_pred'])
        
    BS = brier_score_loss(test_['BAD'], test_['BAD_pred'])


    

    # Round probabilites to classes
    test_['BAD_pred'].loc[test_['BAD_pred'] >= threshold_] = 1
    test_['BAD_pred'].loc[test_['BAD_pred'] < threshold_] = 0
    test_['BAD_pred'] = test_['BAD_pred'].astype(int)
    
    
    # Assess Cost
    Cost = cost_loss(test_['BAD'], test_['BAD_pred'], cost_matrix_test_)
    metrics=[AUC, PAUC, BS, Cost, threshold_]
    
    # Set random seed
    np.random.seed(0)

    # Randomly sample rejected applications of sample size of 20% of accepted applications and label them as BAD
    mask = test_['BAD_pred'] == 1
    sample_size = int(np.mean(start['BAD']) * len(test_[test_['BAD_pred'] ==0]))
    sample_size=min(sample_size, len(test_[test_['BAD_pred'] ==1]))
    ri_customers = np.random.choice(test_[mask].index, size=sample_size, replace=False)

    # Save RI applications
    test_.loc[ri_customers, 'BAD'] = 1
    RI = test_.loc[ri_customers].drop(columns='BAD_pred')

    # Concat accepted applications and RI applications to training data
    goods_pred = test_[test_['BAD_pred']==0].drop(columns='BAD_pred')
    new_data = pd.concat([start_, goods_pred, RI])

    return new_data, metrics


def row_normalized(input_array):
    '''Normalize probabilities to yield a sum of 1.'''
    row_sums = np.sum(input_array, axis=1)
    output_array = input_array / row_sums[:, np.newaxis]
    return output_array



def icp(start, start_trans, test, test_trans_, threshold_, cost_matrix_, cost_matrix_test_, error_func):
    '''Active Learning strategy of using an Inductive Conformal Predictor for assessing Confidence of predicted BADs.
    Reqires and error function of the class type to be put in for the error_func parameter.'''
    
    # Copies of used dataframes
    start_= start.copy()
    start_trans_ = start_trans.copy()
    test_ = test.copy()
    test_trans = test_trans_.copy()
    
    # Split the transformed training data into training and calibration data
    start_trans_split, cal_trans = train_test_split(
        start_trans, test_size=0.2, random_state=42)
    
    # Underlying model
    LR = LogisticRegression(penalty='l1', solver='saga', random_state=0, class_weight='balanced')
    adapter = ClassifierAdapter(LR)
    nc = ClassifierNc(adapter, error_func)
    
    # Inductive Classifier
    icp = IcpClassifier(nc, smoothing=False)
    icp.fit(start_trans_split.drop(columns='BAD').values, start_trans_split['BAD'].values)
    icp.calibrate(cal_trans.drop(columns='BAD').values, cal_trans['BAD'].values)

    # Predict probabilities and normalize them
    preds =icp.predict(test_trans.drop(columns='BAD').values)
    preds = row_normalized(preds)
    
    test_['BAD_pred'] = preds[:, 1]
    
    test_['BAD'] = test_['BAD'].astype(int)
    
    # Assess perfomance
    AUC = roc_auc_score(test_['BAD'],test_['BAD_pred'])
    
    ptest = test_[test_['BAD_pred']<0.5]
    #PAUC = roc_auc_score(ptest['BAD'],ptest['BAD_pred'])
    if len(ptest) == 0 or len(ptest['BAD'].unique()) ==1:
        PAUC = None
    else:
        # calculate AUC score
        PAUC = roc_auc_score(ptest['BAD'],ptest['BAD_pred'])
        
    BS = brier_score_loss(test_['BAD'], test_['BAD_pred'])


    
    
    # Round probabilites to classes
    test_['BAD_pred'].loc[test_['BAD_pred'] >= threshold_] = 1
    test_['BAD_pred'].loc[test_['BAD_pred'] < threshold_] = 0
    test_['BAD_pred'] = test_['BAD_pred'].astype(int)
    
    # Assess confidence of predictions
    test_['pred_conf'] = icp.predict_conf(test_trans.drop(columns='BAD').values)[:, 1]

    # Sample the least confident rejected applications of sample size of the BAD share of accepted applications and turn their prediction
    # from BAD to GOOD
    sample_size = int(np.mean(start['BAD']) * len(test_[test_['BAD_pred'] ==0]))
    sample_size=min(sample_size, len(test_[test_['BAD_pred'] ==1]))
    
    rows_to_switch = test_[test_['BAD_pred'] == 1].nsmallest(sample_size, 'pred_conf').index

    # Switch the selected rows from 0s to 1s in the binary column
    test_.loc[rows_to_switch, 'BAD_pred'] = 0
    
    # Assess Cost
    Cost = cost_loss(test_['BAD'], test_['BAD_pred'], cost_matrix_test_)
    metrics=[AUC, PAUC, BS, Cost, threshold_]

    # Concat accpeted applications to training data
    goods_pred = test_[test_['BAD_pred']==0].drop(columns=['BAD_pred', 'pred_conf'])
    new_data = pd.concat([start_, goods_pred])

    return new_data, metrics


class NearestNeighbourMargin(ClassificationErrFunc):
    '''Nearest Neighbour Margin Error Function. Combines the ideas of Margin Error Function and Nearest Neighbour prediction'''

    def __init__(self):
        super(NearestNeighbourMargin, self).__init__()

    def apply(self, prediction, y):
        prob = np.zeros(y.size, dtype=np.float32)
        ratios=[]
        for i, y_ in enumerate(y):
            if y_ >= prediction.shape[1]:
                prob[i] = 0
            else:        
                prob[i] = prediction[i, int(y_)]
                # distance to nearest neighbour with different class
                diff_neigh= np.min(np.absolute(prob[i] - prediction[:,int(1-y_)]))
                # distance to nearest neighbour with same class
                same_neigh= np.min(np.absolute(prob[i] - prediction[:,int(y_)]))
                ratio = same_neigh / diff_neigh 
                ratios.append(ratio)
        return np.array(ratios)

    
def icp_prob (start, start_trans, test, test_trans_, threshold_, cost_matrix_, cost_matrix_test_):
    '''Active Learning strategy of using an Inductive Conformal Predictor for assessing Confidence of predicted BADs.
    Inverse Probability Error Function used as Error Function'''
    
    return icp(start, start_trans, test, test_trans_, threshold_, cost_matrix_, cost_matrix_test_, InverseProbabilityErrFunc())
 
def icp_nnmargin(start, start_trans, test, test_trans_, threshold_, cost_matrix_, cost_matrix_test_):
    '''Active Learning strategy of using an Inductive Conformal Predictor for assessing Confidence of predicted BADs.
    Nearest Neighbour Margin Error Function used as Error Function''' 
    return icp(start, start_trans, test, test_trans_, threshold_, cost_matrix_, cost_matrix_test_, NearestNeighbourMargin())


