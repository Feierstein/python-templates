import sys
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import boto3_rds_pandas as mysql
import joblib
#setup logging for algorithm
algo_id =  '1715007944_Logistic_Regression'
user_id =  '33u8913nffjo512gss11'
type_ = 'logistical regression'

def test_curl():
    #print('test print in test curl')
    return 'testing curl in sklearn_use_algo'


def add_columns_if_not_exist(df, columns):
    for column in columns:
        if column not in df.columns:
            df[column] = None  # or any default value you want to set
    return df
    
def remove_columns_if_exist(df, columns):
    return df.drop(columns=columns, errors='ignore')
    
def remove_columns_not_in_list(df, columns_to_keep):
    columns_to_remove = [col for col in df.columns if col not in columns_to_keep]
    return df.drop(columns=columns_to_remove)

def reorder_columns(df, column_order):
    return df.reindex(columns=column_order)
    

def run_algo(user_id = '33u8913nffjo512gss11' ):
    print("in sklearn_use_algorithm.get_algo")
    # get algorithm data based off algo_id
    query = f"SELECT * FROM stats.ml_algorithms where id = '{algo_id}';"
    algo_data = mysql.run_sql_query(query)
    
    print('algo_data')
    sql_used = (algo_data["sql_used"][0])
    y_col = (algo_data["y_col"][0])
    type_ = (algo_data["type"][0])
    object_fields_converted = (algo_data["object_fields_converted"][0])
    object_fields_converted = object_fields_converted.split()
       
    index = sql_used.find("WHERE")
    try:
        if(index > 1):
            print(index)
    except Exception as e:
        # Handle any other exceptions
        print("An error occurred there is no 'WHERE' in sql_used: ", e)
        raise CustomError("An error occurred there is no 'WHERE' in sql_used")
    # Split the string at the specified index
    part1 = sql_used[:index]
    part2 = sql_used[index:]
    sql_user = f"{part1} where transaction_id = '{user_id}' " 
    #print('sql_user',sql_user)
    # convert known problem numbers stored as objects
    df = mysql.run_sql_query(sql_user)
    
    # df['months'] = df['months'].astype(float)
    # df['complete_sales_a'] = df['complete_sales_a'].astype(float)
        #clean converted fields and convert them
    for col in object_fields_converted:
        col = str(col)
        col = col.replace("]", "").replace("[", "")
        print('********col',col)
        df[col] = df[col].astype(float)
    #one-hot encoding
    one_hot_encode_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
    one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
    #print('one_hot_encode_cols',one_hot_encode_cols)
    # Encode these columns as categoricals so one hot encoding works on split data (if desired)
    for col in one_hot_encode_cols:
        df[col] = pd.Categorical(df[col])
    # Do the one hot encoding
    df = pd.get_dummies(df, columns=one_hot_encode_cols)
    
    
    #load model based off file name
    loaded_transformation = joblib.load(f'{algo_id}_fit_transform.joblib')
    ## THESE PARAMETERS MUST BE PRESENT IN the test data : parameters_t
    parameters_t = loaded_transformation.get_feature_names_out()
    
    # Add columns if they don't exist
    df = add_columns_if_not_exist(df, parameters_t)
    
    #load model based off file name
    loaded_algo = joblib.load(f'{algo_id}.joblib')
    parameters = loaded_algo.get_params()
    
    # drop y_col to remove the determinate variable
    if y_col in df.columns:
        X = df.drop(y_col, axis=1)
    else:
        X=df
    
    # List of columns to keep
    columns_to_keep = parameters_t
    
    # Remove columns not in the list
    X = remove_columns_not_in_list(X, columns_to_keep)
    
    # Reorder the columns
    X = reorder_columns(X, parameters_t)
    
    # run transformation model
    X_transformed = loaded_transformation.transform(X)
    
    #remove NAN
    X_transformed = np.nan_to_num(X_transformed, nan=0)

    #make prediction
    pred = loaded_algo.predict(X_transformed)
    pred = pred[0]
    pred = int(pred)
    #print('pred',type(pred))



    return_object = {}
    return_object.update({'prediction': pred, 'user_id': user_id, 'algo_id' : algo_id, 'type_' : type_ })
    #return_object['prediction'] = pred
    #pred_string = np.array2string(pred)
    pred_string = return_object
    print("pred_string", pred_string)
    
    return pred_string
    
    #return 'test complete'

def run_algo_on_all(user_id = '33u8913nffjo512gss11' ):
    print("in sklearn _use_algorithm.run_algo_on_all")
    # get algorithm data based off algo_id
    query = f"SELECT * FROM stats.ml_algorithms where id = '{algo_id}';"
    #print("query",query)
    
    algo_data = mysql.run_sql_query(query)
    # print("algo_data",algo_data)
    #print('algo_data')
    sql_used = (algo_data["sql_used"][0])
    y_col = (algo_data["y_col"][0])
    type_ = (algo_data["type"][0])
    object_fields_converted = (algo_data["object_fields_converted"][0])
    object_fields_converted = object_fields_converted.split()
    
    # print("sql_used",sql_used)
    # print("finding the location of where")
    index = sql_used.find("WHERE")
    try:
        if(index > 1):
            print(index)
    except Exception as e:
        # Handle any other exceptions
        print("An error occurred there is no 'WHERE' in sql_used: ", e)
        raise CustomError("An error occurred there is no 'WHERE' in sql_used")
    # Split the string at the specified index
    part1 = sql_used[:index]
    part2 = sql_used[index:]

    sql_user = f"{part1} where data_type = 'U' limit 5 " 
    #print('sql_user',sql_user)
    # convert known problem numbers stored as objects
    df = mysql.run_sql_query(sql_user)
    #print('df',df['transaction_id'])
    user_ids=df['transaction_id']
    
    #clean converted fields and convert them
    for col in object_fields_converted:
        col = str(col)
        col = col.replace("]", "").replace("[", "")
        #print('********col',col)
        df[col] = df[col].astype(float)
    
    
    #one-hot encoding
    one_hot_encode_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
    one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
    #print('one_hot_encode_cols',one_hot_encode_cols)
    # Encode these columns as categoricals so one hot encoding works on split data (if desired)
    for col in one_hot_encode_cols:
        df[col] = pd.Categorical(df[col])
    # Do the one hot encoding
    df = pd.get_dummies(df, columns=one_hot_encode_cols)
    
    #load model based off file name
    loaded_transformation = joblib.load(f'{algo_id}_fit_transform.joblib')
    ## THESE PARAMETERS MUST BE PRESENT IN the test data : parameters_t
    parameters_t = loaded_transformation.get_feature_names_out()
    
    # Add columns if they don't exist
    df = add_columns_if_not_exist(df, parameters_t)
    
    #load model based off file name
    loaded_algo = joblib.load(f'{algo_id}.joblib')
    parameters = loaded_algo.get_params()
    
    # drop y_col to remove the determinate variable
    if y_col in df.columns:
        X = df.drop(y_col, axis=1)
    else:
        X=df
    
    # List of columns to keep
    columns_to_keep = parameters_t
    
    # Remove columns not in the list
    X = remove_columns_not_in_list(X, columns_to_keep)
    
    # Reorder the columns
    X = reorder_columns(X, parameters_t)
    
    # run transformation model
    X_transformed = loaded_transformation.transform(X)
    #print('predictors_t: ', parameters_t )
    #print('X_transformed: ', X_transformed )
    #remove NAN
    X_transformed = np.nan_to_num(X_transformed, nan=0)
    #print('X_transformed: ', X_transformed )
    #make prediction
    pred = loaded_algo.predict(X_transformed)
    
    #good example of 'list comprehension'
    pred = [int(x) for x in pred]

    #pred = int(pred)
    #print("pred", pred)
    #return_object = {}
    #return_object.update({ 'algo_id' : algo_id, 'type_' : type_ })
    #return_object['prediction'] = pred
    #pred_string = np.array2string(pred)
    return_object = {k:v for k,v in zip(user_ids,pred)}
    return_object.update({ 'algo_id' : algo_id, 'type_' : type_ })
    
    pred_string = return_object
    #print("pred_string", pred_string)
    #print(pred_string)
    
    
    return pred_string
    
    
    
    
if __name__ == "__main__":
    run_algo_on_all()
    
    
# def run_algo_on_all(user_id = '33u8913nffjo512gss11' ):
#     print("in sklearn _use_algorithm.run_algo_on_all")
#     # get algorithm data based off algo_id
#     query = f"SELECT * FROM stats.ml_algorithms where id = '{algo_id}';"
#     print("query",query)
    
#     algo_data = mysql.run_sql_query(query)
#     # print("algo_data",algo_data)
#     #print('algo_data')
#     sql_used = (algo_data["sql_used"][0])
#     y_col = (algo_data["y_col"][0])
#     type_ = (algo_data["type"][0])
#     object_fields_converted = (algo_data["object_fields_converted"][0])
#     # print("sql_used",sql_used)
#     # print("finding the location of where")
#     index = sql_used.find("WHERE")
#     try:
#         if(index > 1):
#             print(index)
#     except Exception as e:
#         # Handle any other exceptions
#         print("An error occurred there is no 'WHERE' in sql_used: ", e)
#         raise CustomError("An error occurred there is no 'WHERE' in sql_used")
#     # Split the string at the specified index
#     part1 = sql_used[:index]
#     part2 = sql_used[index:]
#     #print('part1',part1)
#     #sql_user = f"{part1} where transaction_id = '{user_id}' " 
#     sql_user = f"{part1} where data_type = 'U' limit 50 " 
#     print('sql_user',sql_user)
#     # convert known problem numbers stored as objects
#     df = mysql.run_sql_query(sql_user)
#     #print('df',df['transaction_id'])
#     user_ids=df['transaction_id']
#     # df['months'] = df['months'].astype(float)
#     # df['complete_sales_a'] = df['complete_sales_a'].astype(float)
    
    
    
    
    
#     #one-hot encoding
#     one_hot_encode_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
#     one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields
#     #print('one_hot_encode_cols',one_hot_encode_cols)
#     # Encode these columns as categoricals so one hot encoding works on split data (if desired)
#     for col in one_hot_encode_cols:
#         df[col] = pd.Categorical(df[col])
#     # Do the one hot encoding
#     df = pd.get_dummies(df, columns=one_hot_encode_cols)
    
    
#     #load model based off file name
#     loaded_transformation = joblib.load(f'{algo_id}_fit_transform.joblib')
#     ## THESE PARAMETERS MUST BE PRESENT IN the test data : parameters_t
#     parameters_t = loaded_transformation.get_feature_names_out()
    
#     # Add columns if they don't exist
#     df = add_columns_if_not_exist(df, parameters_t)
    
#     #load model based off file name
#     loaded_algo = joblib.load(f'{algo_id}.joblib')
#     parameters = loaded_algo.get_params()
    
#     # drop y_col to remove the determinate variable
#     if y_col in df.columns:
#         X = df.drop(y_col, axis=1)
#     else:
#         X=df
    
#     # List of columns to keep
#     columns_to_keep = parameters_t
    
#     # Remove columns not in the list
#     X = remove_columns_not_in_list(X, columns_to_keep)
    
#     # Reorder the columns
#     X = reorder_columns(X, parameters_t)
    
#     # run transformation model
#     X_transformed = loaded_transformation.transform(X)
#     #print('predictors_t: ', parameters_t )
#     #print('X_transformed: ', X_transformed )
#     #remove NAN
#     X_transformed = np.nan_to_num(X_transformed, nan=0)
#     #print('X_transformed: ', X_transformed )
#     #make prediction
#     pred = loaded_algo.predict(X_transformed)
#     #print("pred", pred)
#     #return_object = {}
#     #return_object.update({ 'algo_id' : algo_id, 'type_' : type_ })
#     #return_object['prediction'] = pred
#     #pred_string = np.array2string(pred)
#     return_object = {k:v for k,v in zip(user_ids,pred)}
#     return_object.update({ 'algo_id' : algo_id, 'type_' : type_ })
    
#     pred_string = return_object
#     #print("pred_string", pred_string)
#     #print(pred_string)
#     return pred_string
    
    



print("sklearn_use_algorithm imported")

sys.stdout.flush()


