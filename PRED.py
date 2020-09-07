#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:06:21 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import category_encoders as ce

# Train Labels
train_y = pd.read_pickle('./data/train_y.pkl')

#%% 1.0

# Get Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_1_0.pkl')
test_x = pd.read_pickle('./data/test_x_enc_1_0.pkl')

# Fit the model and make Prediction
est = LogisticRegression(penalty = 'none', random_state = 1970, solver = 'saga')

model = est.fit(train_x, train_y)
preds = model.predict_proba(test_x)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_1_0.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.73657

#%% 2.0

# Get Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_2_0.pkl')
test_x = pd.read_pickle('./data/test_x_enc_2_0.pkl')

# Fit the model and make Prediction
est = LogisticRegression(penalty = 'none', random_state = 1970, solver = 'saga', 
                         max_iter = 1000)

model = est.fit(train_x, train_y)
preds = model.predict_proba(test_x)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_2_0.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.73726

#%% 2.1.1

# Get Partially Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_2_1_1.pkl')
test_x = pd.read_pickle('./data/test_x_enc_2_1_1.pkl')

ord_cols = [x for x in train_x.columns if 'ord' in x]
te_enc = ce.TargetEncoder(cols = ord_cols)
train_enc = te_enc.fit_transform(train_x, train_y)
test_enc = te_enc.transform(test_x)

est = LogisticRegression(penalty = 'none', random_state = 1970, solver = 'saga', 
                         max_iter = 1000)
model = est.fit(train_enc, train_y)
preds = model.predict_proba(test_enc)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_2_1_1.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.76882

#%% 3.0.1

# Get Partially Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_3_0_1.pkl')
test_x = pd.read_pickle('./data/test_x_enc_3_0_1.pkl')

ord_cols = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0', 'ord_1', 
            'ord_2', 'ord_3', 'ord_4', 'ord_5']
te_enc = ce.TargetEncoder(cols = ord_cols)
train_enc = te_enc.fit_transform(train_x, train_y)
test_enc = te_enc.transform(test_x)

est = LogisticRegression(penalty = 'none', max_iter = 1000, n_jobs = -1, 
                         random_state = 1970, solver = 'saga')
model = est.fit(train_enc, train_y)
preds = model.predict_proba(test_enc)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_3_0_1.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.79237

#%% 3.1.1

# Get Partially Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_3_1_1.pkl')
test_x = pd.read_pickle('./data/test_x_enc_3_1_1.pkl')

ord_cols = [x for x in train_x.columns if 'bin' not in x]
te_enc = ce.TargetEncoder(cols = ord_cols)

train_enc = te_enc.fit_transform(train_x, train_y)
test_enc = te_enc.transform(test_x)

est = LogisticRegression(penalty = 'none', max_iter = 1000, n_jobs = -1, 
                         random_state = 1970, solver = 'saga')
model = est.fit(train_enc, train_y)
preds = model.predict_proba(test_enc)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_3_1_1.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.79237

#%% 3.2.1

# Get Label Train data
train_x = pd.read_pickle('./data/train_x_lbl_1.pkl')
test_x = pd.read_pickle('./data/test_x_lbl_1.pkl')

te_enc = ce.TargetEncoder(cols = train_x.columns)

est = LogisticRegression(penalty = 'none', max_iter = 1000, n_jobs = -1, 
                         random_state = 1970, solver = 'saga')
model = est.fit(train_enc, train_y)
preds = model.predict_proba(test_enc)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_3_2_1.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.79237

#%% 4.0
import lightgbm as lgb

# Get Label Train data
train_x = pd.read_pickle('./data/train_x_lbl_1.pkl')
test_x = pd.read_pickle('./data/test_x_lbl_1.pkl')

est = lgb.LGBMClassifier(random_state = 1970)

model = est.fit(train_x, train_y, categorical_feature = 'auto')
preds = model.predict_proba(test_x)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_4_0.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.79237