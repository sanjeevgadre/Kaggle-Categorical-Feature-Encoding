#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 09:18:19 2020

@author: sanjeev
"""

#%% Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Get Data
train_x = pd.read_csv('./data/train.csv')
test_x = pd.read_csv('./data/test.csv')

#%% Train Data Description
print(train_x.describe(include = 'all').T)
print(train_x.isnull().sum())

'''
1. There are 300,000 rows and 25 columns in the train dataset - 1 id, 23 features and one target - and . 
2. 3 Binary variables bin_0, bin_1 and bin_2 are coded as int
3. 2 Binary variables bin_3 and bin_4 are coded as str
4. All 10 nominal variables are coded as str
5. 1 Ordinal variable ord_0 is coded as int
6. The rest 5 ordinal variables are coded as str
7. 2 datetime variables day and month are coded as int
8. Target variable is coded as int
9. No missing or null values

'''
#%% Test Data Description
print(test_x.describe(include = 'all').T)
print(test_x.isnull().sum())

'''
1. There are 200,000 rows and 24 columns. 
2. The column names across the train and test data match.
3. No missing or null values

'''
#%% Converting data types

# Separating the target variable
train_y = train_x['target']
train_x = train_x.iloc[:, :-1]

# Combining train and test datasets
dat = pd.concat([train_x, test_x], axis = 0)
dat.reset_index(inplace = True, drop = True)
train_idx = len(train_x)

# Converting all columns except id, day and month into categorical variables
dat_cols = dat.columns
for col in dat.columns:
    if col not in ['id', 'day', 'month']:
        dat[col] = dat[col].astype('category')
        
#%% Visualising the category value distribution in columns for train and test dataset side by side
len_f = len(train_x)
len_b = len(test_x)

for col in dat.columns:
    if col not in ['id', 'day', 'month']:
        print('Col: %s Par value: %.4f' % (col, 1/dat[col].describe()['unique']))
        foo = dat.loc[dat.index[:train_idx], col].value_counts()/len_f
        bar = dat.loc[dat.index[train_idx:], col].value_counts()/len_b
        df = pd.concat([foo, bar], axis = 1)
        df.columns = ['train', 'test']
        print(df)
        input('Press any key to continue')
        print('')

'''
1. Across most columns the category value distrubutions for the train and test datasets are comparable. 
2. The differences in category value distributions are in columns with large number of categories. However, given that these categories have a large number of categories (denominator), the differences in category value distributions seem negligible.
3. We tentatively conclude that the train and test datasets are derived from the same population.

'''
#%% Visualizing correlation between binary features and target value in train dataset
bin_cols = [x for x in dat_cols if 'bin' in x]
for col in bin_cols:
    foo = pd.crosstab(train_y, dat.loc[dat.index[:train_idx], col], normalize = True,
                      margins = True)
    bar = np.max([foo.iloc[0,0] + foo.iloc[1,1], foo.iloc[0,1] + foo.iloc[1,0]])
    print('Naive classification using %s will give train set prediction accuracy of %.4f' 
          % (col, bar))
    print('')

'''
1. A naive classification using bin_0 would likely give train set prediction accuracy of ~64.5% and likely forms the baseline threshold for any model development

'''


