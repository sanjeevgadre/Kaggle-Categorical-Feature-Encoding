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
1. The 2.0 encoding strategy of using ordinal coding for ordinal variables did show improvement in the quality of the logistic regression model. However the improvement was minor. One hypothesis is that ordinal coding "assumes" that the gap between successive levels of an ordinal variable is the same. This may not be true. We therefore consider replacing ordinal encoding with Target encoding for ordinal variables to investigate if appropriate contrast encoding for ordinal variables result in better model.
2. In this encoding strategy, we use Target encoding for ordinal variables to ascertain if that has a positive impact on the quality of the logistic regression model.
3. We start with label encoded data from the previous encoding exercise.
4. We one-hot-encode all binary and nominal variables with cardinality < 16; the remaining binary and nominal variables will be binary encoded.

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
bin_cols = [x for x in dat.columns if x not in ohe_cols and 'ord' not in x]
ord_cols = [x for x in dat.columns if 'ord' in x]

# Setting up the encoders
ohe_enc = ce.OneHotEncoder(cols = ohe_cols)
bin_enc = ce.BinaryEncoder(cols = bin_cols)
te_enc = ce.TargetEncoder(cols = ord_cols)

# Applying the encoding
dat = ohe_enc.fit_transform(dat)
dat = bin_enc.fit_transform(dat)

train = dat[:train_idx]
test = dat[train_idx:]
train_enc = te_enc.fit_transform(train, train_y)
test_enc = te_enc.transform(test)

#%% Saving the encoded train and test datsets separately
train_enc.to_pickle('./data/train_x_enc_2.1.pkl')
test_enc.to_pickle('./data/test_x_enc_2.1.pkl')
