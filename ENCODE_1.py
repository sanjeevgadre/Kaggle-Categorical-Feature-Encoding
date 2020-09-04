#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 08:35:20 2020

@author: sanjeev
"""

#%% Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

#%% Get Step-1 Processed data
dat = pd.read_pickle('./data/dat.pkl')
train_idx = 300000      # Number of training samples

#%% Encoding the columns

'''
1. One-hot-encoding is the preferred encoding strategy especially if we plan to develop linear predictive models. For the past pass we assume that we are indeed developing linear predictive models.
2. One-hot-encoding all columns can really blow up the dimension of the dataset and hence we choose to one hot encode only those columns with cardinality < 16. These columns are: bin_0, bin_1, bin_2, bin_3, bin_4, nom_0, nom_1, nom_2, nom_3, nom_4, ord_0, ord_1, ord_2, ord_3, day and month. This will increase the total column count by 67 i.e. to 90 
3. For the remaining columns we use the "Binary Encoding" strategy. This will increase the total column count by another 61 i.e. to 158.
4. Given that the train dataset has 300,000 rows, a total column count of 158 should not pose much difficulty in fitting a linear model. However, it must be highlighted that the resulting feature matrix (after encoding) will be very sparse.

5. Before implementing the encoding strategy outlined above, it will be first necessary to encode all category labels to numeric value. To that end we use the LabelEncoder() method from sklearn.preprocessing

'''

#%% "Labeling" Binary features
le = LabelEncoder()

cols = [c for c in dat.columns if 'bin' in c]
dat.loc[:, cols] = dat.loc[:, cols].apply(le.fit_transform)

#%% "Labeling" Ordinal features - I
cols = [c for c in dat.columns if 'ord' in c]
# Get details of categories for each feature
for col in cols:
    print('Column: %s' % col)
    print(dat[col].unique())
    print('')
    
'''
1. For ord_0 - 3 categories should be ordered as 1, 2, 3
2. For ord_1 - 5 categories should be ordered as Novice, Contributor, Expert, Master, Grandmaster
3. For ord_2 - 6 categories should be orderedordered as Freezing, Cold, Warm, Hot, Boiling Hot, Lava Hot
4. For ord_3 - 15 categories should be ordered ordered a, b, c,.....o
5. For ord_4 - 26 categories should be ordered ordered A, B, C,.....Z
6. For ord_5 - 192 categories. Category names are 2 letter combination and each letter either lowercase or uppercase.

For ord_0, ord_3, ord_4 and ord_5 we let the default setting in LabelEncoder to order the Labels.
For ord_1 and ord_2 we set the LabelEncoder order appropriately.

FOR NOW WE ARE ASSUMING THAT THE MAGNITUDE BETWEEN SUCCESSIVE INTERVALS OF THE ORDINAL DATA IS EQUAL. WE WILL WANT TO REVISIT THIS ASSUMPTION LATER.

'''

#%% "Labeling" Ordinal features - II
for col in cols:
    if col == 'ord_1':
        le.fit(['Novice', 'Contributor', 'Expert', 'Master', 'Grandmaster'])
        dat['ord_1'] = le.transform(dat['ord_1'])
    elif col == 'ord_2':
        le.fit(['Freezing', 'Cold', 'Warm', 'Hot', 'Boiling Hot', 'Lava Hot'])
        dat['ord_2'] = le.transform(dat['ord_2'])
    else:
        dat[col] = le.fit_transform(dat[col])
        
#%% "Labeling" Nominal features
cols = [x for x in dat.columns if 'bin' not in x and 'ord' not in x]
dat.loc[:, cols] = dat.loc[:, cols].apply(le.fit_transform)


#%% "Labeling" turns all data types into 'int64'. We reset it to 'category'
dat = dat.astype('category')

# Saving the "labeled" train and test datasets for future use
dat.iloc[:train_idx, :].to_pickle('./data/train_x_lbl_1.pkl')
dat.iloc[train_idx:, :].to_pickle('./data/test_x_lbl_1.pkl')

#%% One-hot-encoding identified columns
ohe_cols = ['bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'ord_0', 'ord_1', 'ord_2', 'ord_3', 'day', 'month']

ohe = ce.OneHotEncoder(cols = ohe_cols)
dat_ohe_enc = ohe.fit_transform(dat[ohe_cols])

#%% Binary-encoding identified columns
bie_cols = [x for x in dat.columns if x not in ohe_cols]

bie = ce.BinaryEncoder(cols = bie_cols)
dat_bie_enc = bie.fit_transform(dat[bie_cols])

#%% Combining the encoded columns
dat = pd.concat([dat_ohe_enc, dat_bie_enc], axis = 1)

#%% Saving the encoded train and test datsets separately
dat.iloc[:train_idx, :].to_pickle('./data/train_x_enc_1.pkl')
dat.iloc[train_idx:, :].to_pickle('./data/test_x_enc_1.pkl')
