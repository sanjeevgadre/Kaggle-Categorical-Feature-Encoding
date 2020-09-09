INTRODUCTION

1. This project is a learning exercise to understand the use and benefit of different encoding strategies for categorical features.
2. The project uses the Kaggle competition Categorical Feature Encoding Challenge - https://www.kaggle.com/c/cat-in-the-dat
3. The project consists of 4 script files:
    a. EDA.py - Exploratory Data Analysis of the train and test datasets.
    b. ENCODE.py - Different encoding strategies are implemented.
    c. EVAL.py - For each encoding strategy, a 10-fold cross validated estimated training error is computed.
    d. PRED.py - For each encoding strategy, relevant predictions are made for the test dataset.
    
DETAILS

4. Across most columns the category value distrubutions for the train and test datasets are comparable. In other words, the train and test datasets are quite similar. This is an important characteristic of this project and possibly has bearing on the learning derived. 
5. The train dataset contains 3 kinds of categorical features - Binary, Ordinal (of varying cardinality) and Nominal (of varying cardinality).
6. In the train dataset, there is a definite skew in the distribution for all ordinal and nominal feature categories of sample count over target values. However for features with cardinality (large number of categories) this skew becomes less pronounced.
7. The following encoding strategies were implemented:
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

8. For each of the encoding stratgies the 10-fold estimated training error was computed for a logistical classification model. The results for estimated test Area-under-ROC were as follows:
    a. 0.7356
    b. 0.7371
    c. 0.7678
    d. 0.7891
    e. 0.7890
    f. 0.7890
9. For each of the encoding strategies, predictions were made for the test dataset and the results of the predictions were obtained by submitting the results to the competition scorer. The results for the test Area-under-ROC were as follows:
    a. 0.73657
    b. 0.73726
    c. 0.76882
    d. 0.79237
    e. 0.79237
    f. 0.79237
10. Based on the results of the encoding strategy, two variations were tried
    g. <i> Same encoding strategy as d
       <ii> Penalized Logistical classification model instead of a un-penalized classification model
    h. <i> Numerically labeled data
       <ii> LightGBM algorithm based classification model
11. The results for estimated test Area-under-ROC for the two variations were as follows:
    g. 0.7891
    h. 0.7851

LEARNING

12. Encoding categorical features as numbers, a number representing membership of a certain level with in the category, is a necessary first step.
13. For binary and nominal features the order in which categories are numbered does not matter; for ordinal features it does. For ordinal features categories should be numbered to reflect their relative ordinal rank.
14. Given that there is no ranking amongst the categories of the binary and nominal features, one hot encoding is the recommended method to encode the information contained in these features. 
15. For features with high cardinality, one-hot-encoding can blow up the dimensionality of the data matrix and hence alternate encoding like binary encoding may be required. However, binary encoding reduces the "information" content of the data as it slots samples in far lesser number of categories than in reality.
16. One hot encoding ordinal feature results in loss of "information" contained in the relative ranking amongst the categories of an ordinal feature. 
17. Simple ordinal encoding of ordinal features assumes that the relative gaps between consecutive levels of the feature are comparable. This might not be true. Alternate encoding strategy like target encoding that uses target label distribution across an ordinal feature's categories dramatically improves the resultant model.
18. For a high cardinality nominal feature it maybe beneficial to treat the nominal feature as an ordinal feature and use target encoding to represent the feature values.
19. The LightGBM classifier may allow for an tree-ensemble based alternative to a logistical classifier that is easy to encode (requires only appropriate feature labeling and no feature encoding). On the flip side, it might also require significant effort for hyperparameter tuning.

