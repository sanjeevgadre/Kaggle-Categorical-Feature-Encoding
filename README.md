INTRODUCTION

1. This project is a learning exercise to understand the use and benefit of different encoding strategies for categorical variables.
2. The project uses the Kaggle competition Categorical Feature Encoding Challenge - https://www.kaggle.com/c/cat-in-the-dat
3. The project consists of 4 script files:
    a. EDA.py - Exploratory Data Analysis of the train and test datasets.
    b. ENCODE.py - Different encoding strategies are implemented.
    c. EVAL.py - For each encoding strategy, a 10-fold cross validated estimated training error is computed.
    d. PRED.py - For each encoding strategy, relevant predictions are made for the test dataset.
    
DETAILS

4. The train dataset contains 3 kinds of categorical variables - Binary, Ordinal (of varying cardinality) and Nominal (of varying cardinality).
5. The following encoding strategies were implemented:
    a. <i> One-hot-encode Binary and low cardinality Ordinal & Nominal features 
       <ii> Binary encode high cardinality Ordinal & Nominal features
       
    b. <i> One-hot-encode Binary and low cardinality Nominal features 
       <ii> Binary encode high cardinality Nominal features 
       <iii> Ordinal encode Ordinal features
       
    c. <i> One-hot-encode Binary and low cardinality Nominal features 
       <ii> Binary encode high cardinality Nominal features 
       <iii> Target encode Ordinal features
       
    d. <i> One-hot-encode Binary and low cardinality Nominal features 
       <ii> Target encode Ordinal and high cardinality Nominal features 
       
    e. <i> One-hot-encode Binary features
       <ii> Target encode all Ordinal and Nominal features
       
    f. <i> Target encode all features

RESULTS

6. For each of the encoding stratgies the 10-fold estimated training error was computed for a logistical classification model. The results for estimated test Area-under-ROC were as follows:
    a. 0.7356
    b. 0.7371
    c. 0.7678
    d. 0.7891
    e. 0.7890
    f. 0.7890
7. For each of the encoding strategies, predictions were made for the test dataset and the results of the predictions were obtained by submitting the results to the competition scorer. The results for the test Area-under-ROC were as follows:
    a. 0.73657
    b. 0.73726
    c. 0.76882
    d. 0.79237
    e. 0.79237
    f. 0.79237
8. Based on the results of the encoding strategy, two variations were tried
    g. <i> Same encoding strategy as d
       <ii> Penalized Logistical classification model instead of a un-penalized classification model
    h. <i> Numerically labeled data
       <ii> LightGBM algorithm based classification model
9. The results for estimated test Area-under-ROC for the two variations were as follows:
    g. 0.7891
    h. 0.7851

LEARNING



