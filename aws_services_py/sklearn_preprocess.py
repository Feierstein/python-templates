
import sys
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#%pylab inline
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_formats = ['retina']")
import uuid
import time
import json
import boto3
import pandas as pd
from io import BytesIO
import seaborn as sns
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import boto3_rds_pandas as mysql
import joblib

print("running sklearn_preprocess.py")
s = StandardScaler()
mms = MinMaxScaler()
sns.set()
def run_query_test():
    query = "SELECT transaction_id, months, transaction_type,  trust_status, complete_sales_a, reconciliations_c,user_has_ssn,resulted_in_reconcilation FROM stats.user_scores_both WHERE 1=1  "
    df = mysql.run_sql_query(query)
    preprocess(df, 'test', 'unk_name', 'algo_name', 'resulted_in_reconcilation')
    
def preprocess(df, algo_object):
    print("running sklearn_preprocess.preprocess")
    for key, value in algo_object.items():
        globals()[key] = value

    data = df.copy() # Keep a copy our original data 
    
    #remove na and ids and other unique columns 
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
    
    #non number columns
    object_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
    object_cols = object_cols.index.tolist() 
    
    #attempt to convert some object columns to numbers if they have over a certain amount of categories
    for col in object_cols:
        objects_converted = []
        unique_values = df[col].nunique()
        if unique_values > 10:
            objects_converted.append(col)
            df[col] = df[col].astype(float)
    
    #can make some custom adjustments to numbers stored as objects here
    # df['months'] = df['months'].astype(float)
    # df['complete_sales_a'] = df['complete_sales_a'].astype(float)
    
    # columns that are numerical
    num_cols = df.dtypes[df.dtypes != object]  # filtering by string categoricals
    num_cols = num_cols.index.tolist() 
    
    #filter y
    num_cols.remove(y_col)
    
    #gathering stats to eliminate outliers
    rows_outliers = []
    min_max_data = {}
    
    # figure out the stats needed to identify outliers
    for col in num_cols:
        min_v = df[col].min()
        max_v = df[col].max()
        mean_v = df[col].mean()
        stdv_v = df[col].std()
        threshold_h = mean_v + 4 * stdv_v  # 
        threshold_l = mean_v - 4 * stdv_v  # 
    
        df = df.drop(df[df[col] > threshold_h].index)
        df = df.drop(df[df[col] < threshold_l].index)
    
    #after outliers removed
    #depricated? now being stored in python model object
    for col in num_cols:
        min_v_clean = df[col].min()
        max_v_clean = df[col].max()
        mean_v_clean = df[col].mean()
        stdv_v_clean = df[col].std()
        min_max_data[col] = [min_v_clean, max_v_clean, mean_v_clean, stdv_v_clean]
    
    #one hot encoded columns
    one_hot_encode_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
    one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
    
    # Encode these columns as categoricals so one hot encoding works on split data (if desired)
    for col in one_hot_encode_cols:
        df[col] = pd.Categorical(df[col])
    
    # Do the one hot encoding
    df = pd.get_dummies(df, columns=one_hot_encode_cols)
    #df.info()
    
    # get column headers as predictors - must remove y column first
    df_x = df.drop(y_col, axis=1)
    predictors = df_x.columns
    
    #split into X and y data for processing
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    
    # fix all nan in X
    # in this case it deletes rows with any nan to clean data 
    X = X.dropna(axis=0)
    print('X',X)
    
    #polynomial transformations
    pf = PolynomialFeatures(degree=2, include_bias=False,)
    X = pf.fit_transform(X)
    print('X',X)
    
    ## split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=72018)
    
    ## data transformations from standardscaler
    X_train_s = mms.fit_transform(X_train)
    X_test_s = mms.transform(X_test) #using transform instead of fit transform so that it uses the training data only for scaling
    
    #save the transformation object to cloud9
    joblib.dump(mms, f'{algo_id}_fit_transform.joblib')
    
    
    #convert the objects converted 
    print('objects_converted',objects_converted)
    objects_converted = str(objects_converted)
    # objects_converted = objects_converted.replace("'", "")
    # objects_converted = objects_converted.replace("'"] "")
    print('objects_converted',objects_converted)
    return X_train_s,X_test_s,y_train,y_test,objects_converted
    
    
def preprocess_existing_transformation(df, algo_object, objects_converted):
    print("running sklearn_preprocess.preprocess_existing_transformation")
    for key, value in algo_object.items():
        globals()[key] = value
    
    data = df.copy() # Keep a copy our original data 
    
    #remove na and ids and other unique columns 
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
    
    #non number columns
    object_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
    object_cols = object_cols.index.tolist() 
    #print(object_cols)
    
    #attempt to convert some object columns to numbers if they have over a certain amount of categories
    for col in objects_converted:
        df[col] = df[col].astype(float)
    #   # print(df[col])
    #   #print(col)
    #   unique_values = df[col].nunique()
    #   if unique_values > 10:
    #       df[col] = df[col].astype(float)
    
    #can make some custom adjustments to numbers stored as objects here
    # df['months'] = df['months'].astype(float)
    # df['complete_sales_a'] = df['complete_sales_a'].astype(float)
    
    # columns that are numerical
    num_cols = df.dtypes[df.dtypes != object]  # filtering by string categoricals
    num_cols = num_cols.index.tolist() 
    #filter y
    num_cols.remove(y_col)
    
    #gathering stats to eliminate outliers
    rows_outliers = []
    min_max_data = {}
    #print("Number of rows before outliers removed: ",df.shape[0])
    for col in num_cols:
        # figure out the stats needed to identify outliers
        min_v = df[col].min()
        max_v = df[col].max()
        mean_v = df[col].mean()
        stdv_v = df[col].std()
        threshold_h = mean_v + 4 * stdv_v  # 
        threshold_l = mean_v - 4 * stdv_v  # 
    
        df = df.drop(df[df[col] > threshold_h].index)
        df = df.drop(df[df[col] < threshold_l].index)
    
    #print("Number of rows after outliers removed: ",df.shape[0])
    
    #after outliers removed
    #depricated? now being stored in python model object
    for col in num_cols:
        min_v_clean = df[col].min()
        max_v_clean = df[col].max()
        mean_v_clean = df[col].mean()
        stdv_v_clean = df[col].std()
        min_max_data[col] = [min_v_clean, max_v_clean, mean_v_clean, stdv_v_clean]
    
    #one hot encoded columns
    one_hot_encode_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
    one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
    
    # Encode these columns as categoricals so one hot encoding works on split data (if desired)
    for col in one_hot_encode_cols:
        df[col] = pd.Categorical(df[col])
    
    # Do the one hot encoding
    df = pd.get_dummies(df, columns=one_hot_encode_cols)
    #df.info()
    
    # get column headers as predictors - must remove y column first
    df_x = df.drop(y_col, axis=1)
    predictors = df_x.columns
    
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
    
    ## data transformations from standardscaler
    X_train_s = mms.fit_transform(X_train)
    X_test_s = mms.transform(X_test) #using transform instead of fit transform so that it uses the training data only for scaling
    
    #save the transformation object to cloud9
    joblib.dump(mms, f'{algo_id}_fit_transform.joblib')
    
    return X_train_s,X_test_s,y_train,y_test
    
       
    
    
if __name__ == "__main__":
    run_query_test() 
