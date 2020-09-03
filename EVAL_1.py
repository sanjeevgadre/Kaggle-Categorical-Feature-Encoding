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

#%% Get Encoded Train data
train_x = pd.read_pickle('./data/train_x_enc_1.pkl')
train_y = pd.read_pickle('./data/train_y.pkl')

#%% Use GridSearchCV to establish the best parameters for a logistic regression model
est = LogisticRegression(random_state = 1970, solver = 'saga')

cv_scores = cross_val_score(estimator = est, X = train_x, y = train_y, scoring = 'roc_auc', 
                            cv = 10, n_jobs = -1, pre_dispatch = '1.5*n_jobs', 
                            error_score = 'raise')

print('The average cross validated area under the ROC curve %.4f' % np.mean(cv_scores))

# The average cross validated area under the ROC curve 0.7356