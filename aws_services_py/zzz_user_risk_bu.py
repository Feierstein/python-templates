


import sys

# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#%pylab inline
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['retina']")
import inspect
import boto3
import pandas as pd
from io import BytesIO
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
sns.set()


s3 = boto3.resource('s3')

## Load in the transaction level data, this is flexible and can include any field 
bucket_name = 'sagemaker-data-feierstein'
file_name = 'user_score_both_all_fields_20230516.csv'
obj = s3.Object(bucket_name, file_name)
body = obj.get()['Body'].read()
df = pd.read_csv(BytesIO(body))
## df = pd.read_csv("user_scores_backtest.csv") #df = pandas dataframe
## df = pd.read_csv("https://www.dropbox.com/s/ieqco06hryopg59/user_scores_backtest.csv?dl=1")
## target variable this is flexible and can include any continuous numerical 
y_col = "resulted_in_reconcilation" ## known values of whether or not a transaction resulted in a reconciliation


print("Number of rows in the data:", df.shape[0])
print("Number of columns in the data:", df.shape[1])
data = df.copy() # Keep a copy our original data 

df.nunique(axis=0)

# columns that are numerical
num_cols = df.select_dtypes('number').columns

#remove  ids and other unique columns 
df.dropna(how='all', axis=1, inplace=True)#inplace makes it a copy of itself

#drop columns where every value is unique and the type is object
all_rows = len(df)
for col in df.columns:
    if len(df[col].unique()) == all_rows :
        if df[col].dtypes == object:
            df.drop(col,inplace=True,axis=1)
df.nunique(axis=0)

# eliminate rows with nan values before one hot encoding
df = df.dropna(axis=0)

# Get a Pd.Series consisting of all the string categoricals
one_hot_encode_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
#one_hot_encode_cols

# Encode these columns as categoricals so one hot encoding works on split data (if desired)
for col in one_hot_encode_cols:
    df[col] = pd.Categorical(df[col])

# Do the one hot encoding
df = pd.get_dummies(data, columns=one_hot_encode_cols)
#df.info()

# get column headers as predictors - must remove y column first
df_x = df.drop(y_col, axis=1)
predictors = df_x.columns
#print(predictors)

#split into X and y data for processing
X = df.drop(y_col, axis=1)
y = df[y_col]

# fix all nan in X
# in this case it deletes rows with any nan to clean data 
X = X.dropna(axis=0)

#polynomial transformations
# pf = PolynomialFeatures(degree=2, include_bias=False,)
# X = pf.fit_transform(X)

## split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=72018)

## data transformations
X_train_s = s.fit_transform(X_train)
X_test_s = s.fit_transform(X_test)

## LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
lr = LogisticRegression()
lr_result = lr.fit(X_train_s,y_train)
pred = lr.predict(X_test_s)
predictors = df_x.columns
coefficients = lr.coef_
##print(predictors)
##print(coefficients)
mydict = {k:v for k,v in zip(predictors,lr.coef_[0])}
#mydict
print(dir(lr))
#print(mydict)
print("inspect.getmembers(lr)")
print(inspect.getmembers(lr))
#print(lr.solver)



#
#
'''
# start of test on user level data
## Load in the transaction level data, this is flexible and can include any field 
#df2 = pd.read_csv("user_scores.csv") #df2 = pandas dataframe
df2 = pd.read_csv("https://www.dropbox.com/s/nwmjzgbtbp7a8jd/user_scores.csv?dl=1")
data2 = df2.copy() # Keep a copy our original data 
df2.nunique(axis=0)
#print(df2)
# columns that are numerical
num_cols = df2.select_dtypes('number').columns
#num_cols
#remove  ids and other unique columns 
df2.dropna(how='all', axis=1, inplace=True)#inplace makes it a copy of itself
#drop columns where every value is unique and the type is object
all_rows = len(df2)
for col in df2.columns:
    if len(df2[col].unique()) == all_rows :
        if df2[col].dtypes == object:
            df2.drop(col,inplace=True,axis=1)
df2.nunique(axis=0)
# eliminate rows with nan values before one hot encoding
df2 = df2.dropna(axis=0)

# Get a Pd.Series consisting of all the string categoricals
one_hot_encode_cols = df2.dtypes[df2.dtypes == object]  # filtering by string categoricals
one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
#one_hot_encode_cols
# Encode these columns as categoricals so one hot encoding works on split data2 (if desired)
for col in one_hot_encode_cols:
    df2[col] = pd.Categorical(df2[col])
# Do the one hot encoding
df2 = pd.get_dummies(data2, columns=one_hot_encode_cols)
#split into X and y data2 for processing
X2 = df2.drop(y_col, axis=1)
# fix all nan in X
# in this case it deletes rows with any nan to clean data2 
#print(X2.shape)
X2 = X2.dropna(axis=0)
print(X2.shape)
#print(X2.info())
#polynomial transformations
# pf2 = PolynomialFeatures(degree=2, include_bias=False,)
# X2 = pf2.fit_transform(X2)

X2_users = s.transform(X2)

pred2 = lr.predict(X2_users)
print(pred2)
#create a useable list of coeficients
mydict = {k:v for k,v in zip(predictors,lr.coef_[0])}
#mydict
#print(dir(lr_result))

# #print(lr.C)
# # print(lr.__class__)
# # print(lr.__delattr__)
# # print(lr.__dict__)
# # print(lr.__dir__)
# # print(lr.__doc__)
# # print(lr.__eq__)
# # print(lr.__format__)
# # print(lr.__ge__)
# # print(lr.__getattribute__)
# # print(lr.__getstate__)
# # print(lr.__gt__)
# # print(lr.__hash__)
# # print(lr.__init__)
# # print(lr.__init_subclass__)
# # print(lr.__le__)
# # print(lr.__lt__)
# # print(lr.__module__)
# # print(lr.__ne__)
# # print(lr.__new__)
# # print(lr.__reduce__)
# # print(lr.__reduce_ex__)
# # print(lr.__repr__)
# # print(lr.__setattr__)
# # print(lr.__setstate__)
# # print(lr.__sizeof__)
# # print(lr.__str__)
# # print(lr.__subclasshook__)
# # print(lr.__weakref__)
# # print(lr._check_feature_names)
# # print(lr._check_n_features)
# # print(lr._estimator_type)
# # print(lr._get_param_names)
# # print(lr._get_tags)
# # print(lr._more_tags)
# # print(lr._predict_proba_lr)
# # #print(lr._repr_html_)
# # print(lr._repr_html_inner)
# # print(lr._repr_mimebundle_)
# # print(lr._validate_data)
# # print(lr.class_weight)
# # print(lr.classes_)
# # print(lr.coef_)
# print(lr.decision_function())
# print(lr.densify)
# print(lr.dual)
# print(lr.fit)
# print(lr.fit_intercept)
# print(lr.get_params)
# print(lr.intercept_)
# print(lr.intercept_scaling)
# print(lr.l1_ratio)
# print(lr.max_iter)
# print(lr.multi_class)
# print(lr.n_features_in_)
# print(lr.n_iter_)
# print(lr.n_jobs)
# print(lr.penalty)
# print(lr.predict)
# print(lr.predict_log_proba)
# print(lr.predict_proba)
# print(lr.random_state)
# print(lr.score)
# print(lr.set_params)
# print(lr.solver)
# print(lr.sparsify)
# print(lr.tol)
# print(lr.verbose)
# print(lr.warm_start)
'''

sys.stdout.flush()