#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:35:20 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
import category_encoders as ce

#%% Contrast encoding the cardinal columns

'''
1. We extend the 3.0 encoding strategy and now encode all variables, including the binary variables, using the Target Encoding. 
2. We start with label encoded data from the previous encoding exercise.

'''

#%% Get Step-1 Labeled data
dat = pd.concat([pd.read_pickle('./data/train_x_lbl_1.pkl'), 
                 pd.read_pickle('./data/test_x_lbl_1.pkl')], axis = 0)
train_y = pd.read_pickle('./data/train_y.pkl')

train_idx = 300000      # Number of training samples

#%% Category Encoding

# Setting up the encoders
te_enc = ce.TargetEncoder(cols = dat.columns)

# Applying the encoding

train = dat[:train_idx]
test = dat[train_idx:]
train_enc = te_enc.fit_transform(train, train_y)
test_enc = te_enc.transform(test)

#%% Saving the encoded train and test datsets separately
train_enc.to_pickle('./data/train_x_enc_3.2.pkl')
test_enc.to_pickle('./data/test_x_enc_3.2.pkl')
