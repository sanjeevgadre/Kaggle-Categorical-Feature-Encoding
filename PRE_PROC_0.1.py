#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:35:20 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

#%% Get Step-1 Processed data
dat = pd.read_pickle('./data/dat_01.pkl')
train_y = pd.read_pickle('./data/train_y_01.pkl')
train_idx = 300000      # Number of training samples

#%% Labeling the data
'''
1. "Binarize" the binary features.
2. "One-hot-encode" the nominal features.
3. "Label" the ordinal features.

4. For 1 and 3, we use sklearn.preprocessing.LabelEncoder(). However it is important to note that the encoding order for individual ordinal variable is carefully set to preserve the order amongst the categories

'''
# For Binary features
cols = [c for c in dat.columns if 'bin' in c]
dat.loc[:, cols] = dat.loc[:, cols].apply(LabelEncoder().fit_transform)

#%% For Ordinal features
cols = [c for c in dat.columns if 'ord' in c]
# Get details of categories for each feature
for col in cols:
    print('Column: %s' % col)
    print(dat[col].unique())
    print('')
    
'''
1. For ord_0 - 3 categories ordered as 1, 2, 3
2. For ord_1 - 5 categories ordered as Novice, Contributor, Expert, Master, Grandmaster
3. For ord_2 - 6 categories ordered as Freezing, Cold, Warm, Hot, Boiling Hot, Lava Hot
4. For ord_3 - 15 categories ordered a, b, c,.....o
5. For ord_4 - 26 categories ordered A, B, C,.....Z
6. For ord_5 - 192 categories. Category names are 2 letter combination and each letter either lowercase or uppercase.

For ord_0, ord_3, ord_4 and ord_5 we let the default setting in LabelEncoder to order the Labels.
For ord_1 and ord_2 we set the LabelEncoder order appropriately.

FOR NOW WE ARE ASSUMING THAT THE MAGNITUDE BETWEEN SUCCESSIVE INTERVALS OF THE ORDINAL DATA IS EQUAL. WE WILL WANT TO REVISIT THIS ASSUMPTION LATER.

'''
le = LabelEncoder()
for col in cols:
    if col == 'ord_1':
        le.fit(['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'])
        dat['ord_1'] = le.transform(dat['ord_1'])
    elif col == 'ord_2':
        le.fit(['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'])
        dat['ord_2'] = le.transform(dat['ord_2'])
    else:
        dat[col] = le.fit_transform(dat[col])

#%% Saving the processed data for future use
dat.to_pickle('./data/dat_02.pkl')


#%%
foo = dat.loc[:10, :].copy()

