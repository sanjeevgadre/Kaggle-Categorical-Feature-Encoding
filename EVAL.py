#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:06:21 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import category_encoders as ce

# Train Labels
train_y = pd.read_pickle('./data/train_y.pkl')

# Fold variables
n_folds = 10

#%% 1.0 One-hot-encode Binary and low cardinality Ordinal & Nominal features | Binary encode high cardinality Ordinal & Nominal features

# Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_1_0.pkl')

# Evaluate the average cross validated score
est = LogisticRegression(penalty = 'none', random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            n_jobs = -1, pre_dispatch = '1.5*n_jobs', cv = 10, 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7356

#%% 2.0 One-hot-encode Binary and low cardinality Nominal features | Binary encode high cardinality Nominal features | Ordinal Encode Ordinal Features

# Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_2_0.pkl')

# Evaluate the average cross validated score
est = LogisticRegression(penalty = 'none', max_iter = 1000, 
                         random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            n_jobs = -1, pre_dispatch = '1.5*n_jobs', cv = 10, 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7371. The score has improved marginally from the first pass.

#%% 2.1.1 One-hot-encode Binary and low cardinality Nominal features | Binary encode high cardinality Nominal features | Target Encode Ordinal Features

# Get Partially Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_2_1_1.pkl')
train_len = len(train_x)
fold_size = train_len/n_folds

'''
1. We do a 10-fold cross validation exercise.
2. For each pass, we leave out one fold as a validation fold, perform the Target encoding on the rest of the data (using the target labels) and then do the Target encoding on the left-out fold without the target labels

'''
ord_cols = [x for x in train_x.columns if 'ord' in x]
te_enc = ce.TargetEncoder(cols = ord_cols)
est = LogisticRegression(penalty = 'none', max_iter = 1000, 
                         random_state = 1970, solver = 'saga')
cv_scores = []

for fold_id in range(n_folds):
    # Setting up the indices for the validation subset
    val_idx = int(fold_id*fold_size)
    val_idx = np.arange(val_idx, val_idx + fold_size, dtype = 'int')
    
    # Encoding the training and validation subsets separately
    train_enc = te_enc.fit_transform(train_x.loc[~train_x.index.isin(val_idx)], 
                                     train_y.loc[~train_y.index.isin(val_idx)])
    val_enc = te_enc.transform(train_x.loc[val_idx])
    
    # Fit a logistic regression model to the train subset
    model = est.fit(train_enc, train_y.loc[~train_y.index.isin(val_idx)])
    # Predict probability estimates for val subset
    pred_proba = model.predict_proba(val_enc)
    # Calculate the ROC-AUC for the prediction and append it to the list
    cv_scores.append(roc_auc_score(train_y.loc[val_idx], pred_proba[:, 1]))
    
print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7678. The score has improved significantly over the previous pass.

#%% 3.0.1 One-hot-encode Binary and low cardinality Nominal features | Target Encode Ordinal and high cardinality Nominal Features

# Get Partially Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_3_0_1.pkl')
train_len = len(train_x)
fold_size = train_len/n_folds

'''
1. We do a 10-fold cross validation exercise.
2. For each pass, we leave out one fold as a validation fold, perform the Target encoding on the rest of the data (using the target labels) and then do the Target encoding on the left-out fold without the target labels

'''
ord_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0', 'ord_1', 
            'ord_2', 'ord_3', 'ord_4', 'ord_5']
te_enc = ce.TargetEncoder(cols = ord_cols)
est = LogisticRegression(penalty = 'none', max_iter = 1000, n_jobs = -1, 
                         random_state = 1970, solver = 'saga')
cv_scores = []

for fold_id in range(n_folds):
    # Setting up the indices for the validation subset
    val_idx = int(fold_id*fold_size)
    val_idx = np.arange(val_idx, val_idx + fold_size, dtype = 'int')
    
    # Encoding the training and validation subsets separately
    train_enc = te_enc.fit_transform(train_x.loc[~train_x.index.isin(val_idx)], 
                                     train_y.loc[~train_y.index.isin(val_idx)])
    val_enc = te_enc.transform(train_x.loc[val_idx])
    
    # Fit a logistic regression model to the train subset
    model = est.fit(train_enc, train_y.loc[~train_y.index.isin(val_idx)])
    # Predict probability estimates for val subset
    pred_proba = model.predict_proba(val_enc)
    # Calculate the ROC-AUC for the prediction and append it to the list
    cv_scores.append(roc_auc_score(train_y.loc[val_idx], pred_proba[:, 1]))
    
print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7891. The score has improved significantly from the second pass.

#%% 3.0.2 Same coding strategy as 3.0.1 | Changing the Estimator from Simple Logistic Regression to L1-penalized Logistic Regression

# Get Partially Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_3_0_1.pkl')
train_len = len(train_x)
fold_size = train_len/n_folds

'''
1. We do a 10-fold cross validation exercise.
2. For each pass, we leave out one fold as a validation fold, perform the Target encoding on the rest of the data (using the target labels) and then do the Target encoding on the left-out fold without the target labels

'''
ord_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0', 'ord_1', 
            'ord_2', 'ord_3', 'ord_4', 'ord_5']
te_enc = ce.TargetEncoder(cols = ord_cols)
est = LogisticRegression(penalty = 'l1', max_iter = 1000, n_jobs = -1, 
                         random_state = 1970, solver = 'saga')
cv_scores = []

for fold_id in range(n_folds):
    # Setting up the indices for the validation subset
    val_idx = int(fold_id*fold_size)
    val_idx = np.arange(val_idx, val_idx + fold_size, dtype = 'int')
    
    # Encoding the training and validation subsets separately
    train_enc = te_enc.fit_transform(train_x.loc[~train_x.index.isin(val_idx)], 
                                     train_y.loc[~train_y.index.isin(val_idx)])
    val_enc = te_enc.transform(train_x.loc[val_idx])
    
    # Fit a logistic regression model to the train subset
    model = est.fit(train_enc, train_y.loc[~train_y.index.isin(val_idx)])
    # Predict probability estimates for val subset
    pred_proba = model.predict_proba(val_enc)
    # Calculate the ROC-AUC for the prediction and append it to the list
    cv_scores.append(roc_auc_score(train_y.loc[val_idx], pred_proba[:, 1]))
    
print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7891. The score has not improved over the previous pass. The change of estimator has not proven beneficial. However the estimator hyperparameters have not been tuned and it might be worthwile to experiment with tuning to see if there is any improvement.

#%% 3.1.1 One-hot-encode Binary features | Target encode all Ordinal and Nominal features

# Get Partially Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_3_1_1.pkl')
train_len = len(train_x)
fold_size = train_len/n_folds

'''
1. We do a 10-fold cross validation exercise.
2. For each pass, we leave out one fold as a validation fold, perform the Target encoding on the rest of the data (using the target labels) and then do the Target encoding on the left-out fold without the target labels

'''
ord_cols = [x for x in train_x.columns if 'bin' not in x]
te_enc = ce.TargetEncoder(cols = ord_cols)
est = LogisticRegression(penalty = 'none', max_iter = 1000, n_jobs = -1, 
                         random_state = 1970, solver = 'saga')
cv_scores = []

for fold_id in range(n_folds):
    # Setting up the indices for the validation subset
    val_idx = int(fold_id*fold_size)
    val_idx = np.arange(val_idx, val_idx + fold_size, dtype = 'int')
    
    # Encoding the training and validation subsets separately
    train_enc = te_enc.fit_transform(train_x.loc[~train_x.index.isin(val_idx)], 
                                     train_y.loc[~train_y.index.isin(val_idx)])
    val_enc = te_enc.transform(train_x.loc[val_idx])
    
    # Fit a logistic regression model to the train subset
    model = est.fit(train_enc, train_y.loc[~train_y.index.isin(val_idx)])
    # Predict probability estimates for val subset
    pred_proba = model.predict_proba(val_enc)
    # Calculate the ROC-AUC for the prediction and append it to the list
    cv_scores.append(roc_auc_score(train_y.loc[val_idx], pred_proba[:, 1]))
    
print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7890. The score has not improved from the previous pass.

#%% 3.2.1 Target encode all features

# Get Label Train data
train_x = pd.read_pickle('./data/train_x_lbl_1.pkl')
train_len = len(train_x)
fold_size = train_len/n_folds

'''
1. We do a 10-fold cross validation exercise.
2. For each pass, we leave out one fold as a validation fold, perform the Target encoding on the rest of the data (using the target labels) and then do the Target encoding on the left-out fold without the target labels

'''
#ord_cols = [x for x in train_x.columns if 'bin' not in x]
te_enc = ce.TargetEncoder(cols = train_x.columns)
est = LogisticRegression(penalty = 'none', max_iter = 1000, n_jobs = -1, 
                         random_state = 1970, solver = 'saga')
cv_scores = []

for fold_id in range(n_folds):
    # Setting up the indices for the validation subset
    val_idx = int(fold_id*fold_size)
    val_idx = np.arange(val_idx, val_idx + fold_size, dtype = 'int')
    
    # Encoding the training and validation subsets separately
    train_enc = te_enc.fit_transform(train_x.loc[~train_x.index.isin(val_idx)], 
                                     train_y.loc[~train_y.index.isin(val_idx)])
    val_enc = te_enc.transform(train_x.loc[val_idx])
    
    # Fit a logistic regression model to the train subset
    model = est.fit(train_enc, train_y.loc[~train_y.index.isin(val_idx)])
    # Predict probability estimates for val subset
    pred_proba = model.predict_proba(val_enc)
    # Calculate the ROC-AUC for the prediction and append it to the list
    cv_scores.append(roc_auc_score(train_y.loc[val_idx], pred_proba[:, 1]))
    
print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7890. The score has not improved from the previous pass.

#%% 4.0 - Using the LightGBM algorithm
import lightgbm as lgb

# Get Label Train data
train_x = pd.read_pickle('./data/train_x_lbl_1.pkl')

# Evaluate the average cross validated score
est = lgb.LGBMClassifier(random_state = 1970)

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            n_jobs = -1, pre_dispatch = '1.5*n_jobs', cv = 10, 
                            fit_params = {'categorical_feature' : 'auto'}, 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7851. The score has not improved from the previous pass. However LightGBM has multiple parameters to tune and it might be useful to see the performance under "best" parameters
