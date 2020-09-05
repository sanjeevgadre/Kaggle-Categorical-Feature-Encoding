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
1. When we used (in 1.0) one-hot-encoding or binary encoding for ordinal columns, we in effect treat these ordinal variables as cardinal variables and thereby lost the "information carried" in the variable's levels (values)
2. In this encoding strategy, we use ordinal encoding for ordinal variables to ascertain if that has a positive impact on the quality of the logistic regression model.
3. We start with label encoded data from the previous encoding exercise.
4. We one-hot-encode all binary and nominal variables with cardinality < 16; the remaining binary and nominal variables will be binary encoded.

'''

#%% Get Step-1 Labeled data
dat = pd.concat([pd.read_pickle('./data/train_x_lbl_1.pkl'), 
                 pd.read_pickle('./data/test_x_lbl_1.pkl')], axis = 0)

train_idx = 300000      # Number of training samples

#%% Category Encoding

# Identifying columns according to encoding to apply
ohe_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3',
            'nom_4', 'day', 'month']
bin_cols = [x for x in dat.columns if x not in ohe_cols and 'ord' not in x]

# Setting up the encoders
ohe_enc = ce.OneHotEncoder(cols = ohe_cols)
bin_enc = ce.BinaryEncoder(cols = bin_cols)

# Applying the encoding
dat = ohe_enc.fit_transform(dat)
dat = bin_enc.fit_transform(dat)

#%% Saving the encoded train and test datsets separately
dat.iloc[:train_idx, :].to_pickle('./data/train_x_enc_2_0.pkl')
dat.iloc[train_idx:, :].to_pickle('./data/test_x_enc_2_0.pkl')
