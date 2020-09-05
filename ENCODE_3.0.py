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
1. The 2.x encoding strategy of ordinal encoding ordinal variables and ensuring their relative level values are captured resulted in improved performance of the logistic regression model. 
2. In this encoding strategy, we extend the use Target encoding to cardinal variables with cardinality > 16 to ascertain if that has a positive impact on the quality of the logistic regression model. This is in effect treating the cardinal variables as ordinal variables.
3. We start with label encoded data from the previous encoding exercise.
4. We one-hot-encode all binary and nominal variables with cardinality < 16

'''

#%% Get Step-1 Labeled data
dat = pd.concat([pd.read_pickle('./data/train_x_lbl_1.pkl'), 
                 pd.read_pickle('./data/test_x_lbl_1.pkl')], axis = 0)
train_y = pd.read_pickle('./data/train_y.pkl')

train_idx = 300000      # Number of training samples

#%% Category Encoding

# Identifying columns according to encoding to apply
ohe_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3',
            'nom_4', 'day', 'month']
ord_cols = [x for x in dat.columns if x not in ohe_cols]

# Setting up the encoders
ohe_enc = ce.OneHotEncoder(cols = ohe_cols)
te_enc = ce.TargetEncoder(cols = ord_cols)

# Applying the encoding
dat = ohe_enc.fit_transform(dat)

train = dat[:train_idx]
test = dat[train_idx:]
train_enc = te_enc.fit_transform(train, train_y)
test_enc = te_enc.transform(test)

#%% Saving the encoded train and test datsets separately
train_enc.to_pickle('./data/train_x_enc_3.pkl')
test_enc.to_pickle('./data/test_x_enc_3.pkl')
