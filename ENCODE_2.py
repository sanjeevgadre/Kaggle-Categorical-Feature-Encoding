#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:35:20 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
import category_encoders as ce

#%% Contrast encoding the ordinal columns

'''
1. When we use one-hot-encoding or binary encoding for ordinal columns, we in effect treat these ordinal variables as cardinal variables and thereby lose the "information carried" in the variable's levels (values)
2. In this encoding strategy, we use ordinal encoding for ordinal variables to ascertain if that has a positive impact on the quality of the logistic regression model.
3. We start with label encoded data from the previous encoding exercise.
4. We one-hot-encode all binary and nominal variables with cardinality < 16; the remaining binary and nominal variables will be binary encoded.

'''

#%% Get Step-1 Labeled data
dat = pd.concat([pd.read_pickle('./data/train_x_lbl_1.pkl'), 
                 pd.read_pickle('./data/test_x_lbl_1.pkl')], axis = 0)

train_idx = 300000      # Number of training samples

#%% One-hot-encoding identified columns
ohe_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'day', 'month']

ohe = ce.OneHotEncoder(cols = ohe_cols)
dat_ohe_enc = ohe.fit_transform(dat[ohe_cols])

#%% Binary-encoding identified columns
bie_cols = [x for x in dat.columns if x not in ohe_cols and 'ord' not in x]

bie = ce.BinaryEncoder(cols = bie_cols)
dat_bie_enc = bie.fit_transform(dat[bie_cols])

#%% Combining the encoded columns with labeled ordinal columns
ord_cols = [x for x in dat.columns if 'ord' in x]

dat = pd.concat([dat_ohe_enc, dat_bie_enc, dat[ord_cols]], axis = 1)

#%% Saving the encoded train and test datsets separately
dat.iloc[:train_idx, :].to_pickle('./data/train_x_enc_2.pkl')
dat.iloc[train_idx:, :].to_pickle('./data/test_x_enc_2.pkl')
