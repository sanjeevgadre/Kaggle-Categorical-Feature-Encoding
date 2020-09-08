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

#%% 1. One-hot-encode Binary and low cardinality Ordinal & Nominal features | Binary encode high cardinality Ordinal & Nominal features

# Get Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_1.pkl')
test_x = pd.read_pickle('./data/test_x_enc_1.pkl')

# Fit the model and make Prediction
est = LogisticRegression(penalty = 'none', random_state = 1970, solver = 'saga')

model = est.fit(train_x, train_y)
preds = model.predict_proba(test_x)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_1.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.73657

#%% 2. One-hot-encode Binary and low cardinality Nominal features | Binary encode high cardinality Nominal features | Ordinal encode Ordinal features

# Get Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_2.pkl')
test_x = pd.read_pickle('./data/test_x_enc_2.pkl')

# Fit the model and make Prediction
est = LogisticRegression(penalty = 'none', random_state = 1970, solver = 'saga', 
                         max_iter = 1000)

model = est.fit(train_x, train_y)
preds = model.predict_proba(test_x)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_2.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.73726

#%% 3. One-hot-encode Binary and low cardinality Nominal features | Binary encode high cardinality Nominal features | Target encode Ordinal features

# Get Partially Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_3.pkl')
test_x = pd.read_pickle('./data/test_x_enc_3.pkl')

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

preds.to_csv('./data/preds_3.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.76882

#%% 4. One-hot-encode Binary and low cardinality Nominal features | Target Encode Ordinal and high cardinality Nominal features

# Get Partially Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_4.pkl')
test_x = pd.read_pickle('./data/test_x_enc_4.pkl')

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

preds.to_csv('./data/preds_4.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.79237

#%% 5. One-hot-encode Binary features | Target encode all Ordinal and Nominal features

# Get Partially Encoded Train and Test data
train_x = pd.read_pickle('./data/train_x_enc_5.pkl')
test_x = pd.read_pickle('./data/test_x_enc_5.pkl')

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

preds.to_csv('./data/preds_5.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.79237

#%% 6. Target encode all features

# Get Label Train data
train_x = pd.read_pickle('./data/train_x_lbl.pkl')
test_x = pd.read_pickle('./data/test_x_lbl.pkl')

te_enc = ce.TargetEncoder(cols = train_x.columns)

est = LogisticRegression(penalty = 'none', max_iter = 1000, n_jobs = -1, 
                         random_state = 1970, solver = 'saga')
model = est.fit(train_enc, train_y)
preds = model.predict_proba(test_enc)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_6.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.79237

#%% 8. LightGBM Algorithm
import lightgbm as lgb

# Get Label Train data
train_x = pd.read_pickle('./data/train_x_lbl.pkl')
test_x = pd.read_pickle('./data/test_x_lbl.pkl')

est = lgb.LGBMClassifier(random_state = 1970)

model = est.fit(train_x, train_y, categorical_feature = 'auto')
preds = model.predict_proba(test_x)
preds = pd.DataFrame(data = {'id': np.arange(300000, 300000+len(test_x)), 
                             'target': preds[:, 1]}, columns = ['id', 'target'])

preds.to_csv('./data/preds_8.csv', index = False, index_label = False)

# The test set area under the ROC curve 0.79237