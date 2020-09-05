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

# Train Labels
train_y = pd.read_pickle('./data/train_y.pkl')

#%% 1.0

# Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_1.pkl')

# Evaluate the average cross validated score
est = LogisticRegression(random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            cv = 10, n_jobs = -1, pre_dispatch = '1.5*n_jobs', 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7356

#%% 2.0

# Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_2.pkl')

# Evaluate the average cross validated score
est = LogisticRegression(max_iter = 1000, random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            cv = 10, n_jobs = -1, pre_dispatch = '1.5*n_jobs', 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7371. The score has improved from the first pass.

#%% 2.1

# Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_2.1.pkl')

# Evaluate the average cross validated score
est = LogisticRegression(max_iter = 1000, random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            cv = 10, n_jobs = -1, pre_dispatch = '1.5*n_jobs', 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7687. The score has improved from the previous pass.

#%% 3.0

# Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_3.pkl')

# Evaluate the average cross validated score
est = LogisticRegression(max_iter = 1000, random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            cv = 10, n_jobs = -1, pre_dispatch = '1.5*n_jobs', 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.8336. The score has improved significantly from the second pass.

#%% 3.1

# Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_3.1.pkl')

# Evaluate the average cross validated score
est = LogisticRegression(max_iter = 1000, random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            cv = 10, n_jobs = -1, pre_dispatch = '1.5*n_jobs', 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.8337. The score has not improved from the previous pass.

#%% 3.2

# Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_3.2.pkl')

# Evaluate the average cross validated score
est = LogisticRegression(max_iter = 1000, random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            cv = 10, n_jobs = -1, pre_dispatch = '1.5*n_jobs', 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.8336. The score has not improved from the previous pass.