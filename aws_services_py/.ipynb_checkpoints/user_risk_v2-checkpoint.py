


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
#setup logging for algorithm
algo_type = 'Logistical'
algo_id =  str(int(time.time())) + '_' + algo_type
algo_name =  'Logistical Regression user_risk'

print('algo_name')
print(algo_name)

y_col = "resulted_in_reconcilation" ## known values of whether or not a transaction resulted in a reconciliation
show_logs = False

s = StandardScaler()
mms = MinMaxScaler()
sns.set()


query = "SELECT transaction_id, months, transaction_type,  trust_status, complete_sales_a, reconciliations_c,user_has_ssn,resulted_in_reconcilation FROM stats.user_scores_both WHERE 1=1  "
df = mysql.run_sql_query(query)
data = df.copy() # Keep a copy our original data 

query = query.lower()

# add algorithm data to ml_algorithms
insert_data = algo_id
query_insert = f"insert into stats.ml_algorithms (created_at,id,name,type,y_col,sql_used) values(now(),'{algo_id}','{algo_name}','{algo_type}','{y_col}','{query}')"
df2 = mysql.run_sql_insert(query_insert)


# basic data stats
if(show_logs == True):
    print('df')
    print(df)
    print("Number of rows in the raw data:", df.shape[0])
    print("Number of columns in the raw data:", df.shape[1])
    print(df.nunique(axis=0))
    print(df.describe())
    



    #print('df.columns')
    #print(df.columns)


# force certain columns to change from object to float
#cols = cols.index.tolist() 
object_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
object_cols = object_cols.index.tolist() 

# for column in object_cols:
#     print('df[column]')
#     print(df[column])
#     if df[column].dtype == 'object' and df[column].str.match(r'^-?\d+\.?\d*$').any():
#         df[column] = df[column].astype(float)

# print(df)

df['months'] = df['months'].astype(float)
df['complete_sales_a'] = df['complete_sales_a'].astype(float)

# columns that are numerical
num_cols = df.dtypes[df.dtypes != object]  # filtering by string categoricals
num_cols = num_cols.index.tolist() 
#filter y
num_cols.remove(y_col)

#
#NEED TO ADD FILTRATION FOR OUTLIERS HERE
#

# num_cols = df.select_dtypes('number').columns
rows_outliers = []
min_max_data = {}


print("Number of rows before outliers removed: ",df.shape[0])


for col in num_cols:
    # figure out the stats needed to identify outliers
    min_v = df[col].min()
    max_v = df[col].max()
    mean_v = df[col].mean()
    stdv_v = df[col].std()
    threshold_h = mean_v + 3 * stdv_v  # 
    threshold_l = mean_v - 3 * stdv_v  # 
    # Boolean indexing to filter rows
    # rows_outliers += df[df[col] > threshold_h].index.tolist()
    # rows_outliers += df[df[col] < threshold_l].index.tolist()
    # drop outliers
    df = df.drop(df[df[col] > threshold_h].index)
    df = df.drop(df[df[col] < threshold_l].index)

print("Number of rows after outliers removed: ",df.shape[0])

for col in num_cols:
    # figure out the stats needed to identify outliers
    min_v_clean = df[col].min()
    max_v_clean = df[col].max()
    mean_v_clean = df[col].mean()
    stdv_v_clean = df[col].std()
    min_max_data[col] = [min_v_clean, max_v_clean, mean_v_clean, stdv_v_clean]
# print('min_max_data')
# print(min_max_data)


####
#sys.exit(0)#shut down code beyond this point
####


if(show_logs == True):
    print('df.dtypes')
    print(df.dtypes)
    print('num_cols')
    print(num_cols)
    print('df[months]')
    print(df['months'])
    print(pd.Categorical(df[num_cols]))

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

#df_y_filtered_list = [x for x in my_list if not isinstance(x, str)]
# Get a Pd.Series consisting of all the string categoricals
one_hot_encode_cols = df.dtypes[df.dtypes == object]  # filtering by string categoricals
one_hot_encode_cols = one_hot_encode_cols.index.tolist()  # list of categorical fields

# Encode these columns as categoricals so one hot encoding works on split data (if desired)
for col in one_hot_encode_cols:
    df[col] = pd.Categorical(df[col])

# Do the one hot encoding
df = pd.get_dummies(df, columns=one_hot_encode_cols)
#df = pd.get_dummies(data, columns=one_hot_encode_cols)
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
#from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import MinMaxScaler
# X_train_s = s.fit_transform(X_train)
# X_test_s = s.fit_transform(X_test)
X_train_s = mms.fit_transform(X_train)
X_test_s = mms.transform(X_test) #using transform instead of fit transform so that it uses the training data only for scaling
joblib.dump(mms, f'{algo_id}_fit_transform.joblib')
# print("Scale:", mms.scale_)
# print("Min:", mms.min_)
# print("Data Min:", mms.data_min_)
# print("Data Max:", mms.data_max_)
# ###
# sys.exit(0)#shut down code beyond this point
# ###
try:
## LogisticRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn import metrics
    lr = LogisticRegression()
    print('X_train_s',X_train_s)
    print('y_train',y_train)
    
    lr_result = lr.fit(X_train_s,y_train)
    pred = lr.predict(X_test)
    joblib.dump(lr, f'{algo_id}.joblib')
    predictors = df_x.columns
    coefficients = lr.coef_
except Exception as e:
    query_insert4 = f"update stats.ml_algorithms set parameters = 'failed', object = '{e}', coefficients = 'failed' where id ='{algo_id}' "
    df4 = mysql.run_sql_insert(query_insert4)
    print("An error occurred:", e)
## sys.exit(0)#shut down code beyond this point
###

##print(predictors)
##print(coefficients)
mydict = {k:v for k,v in zip(predictors,lr.coef_[0])}

coefficients = lr.coef_
intercept = lr.intercept_

# Accessing other attributes
penalty = lr.penalty
solver = lr.solver
C = lr.C

# Inspecting the learned parameters
parameters = lr.get_params()

# Evaluate the model
accuracy = accuracy_score(y_test, pred)
precision = precision_score(y_test, pred)
recall = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)
roc_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])


# algorithm_info = {
# "Coefficients:" : coefficients,
# "Intercept:" :  intercept,
# "Penalty:" : penalty,
# "Solver:" : solver,
# "Regularization strength (C):": C,
# "Parameters:": parameters,
# "Accuracy" : accuracy,
# "Precision" : precision,
# "Recall": recall,
# "F1" : f1,
# "roc_auc" : roc_auc
# }
# print('algorithm_info')
# print(algorithm_info)
# algorithm_info = str(algorithm_info) 
# algorithm_info = "`"+ algorithm_info + "`"
# print('algorithm_info') 
# print(algorithm_info)
# print('coefficients')
# print(coefficients)
# print('intercept')
# print(intercept)
# print('penalty')
# print(penalty)
# print('solver')
# print(solver)
# print('C')
# print(C)
# print('parameters')
# print(parameters)
# print('accuracy')
# print(accuracy)

#still an issue with this one 
print('precision')
print(precision)

# print('recall')
# print(recall)
# print('f1')
# print(f1)
# print('roc_auc')
# print(roc_auc)
#parameters = f"\"{parameters}\""
# converting python objects to json
parameters = json.dumps(parameters)
precision = json.dumps(precision)
# add algorithm data to ml_algorithms
#query_insert2 = f"update stats.ml_algorithms set coefficients = '{coefficients}', intercept = '{intercept}',	penalty = '{penalty}',	solver = '{solver}',	C = '{C}',	parameters = '{parameters}',	accuracy = '{accuracy}',	precision = '{precision}',	recall = '{recall}',	f1 = '{f1}',	roc_auc = '{roc_auc}'  where id ='{algo_id}' "
query_insert2 = f"update stats.ml_algorithms set parameters = '{parameters}', object = '{lr}', coefficients = '{coefficients}', intercept = '{intercept}',	penalty = '{penalty}',	solver = '{solver}',	C = '{C}',	accuracy = '{accuracy}', 	recall = '{recall}',	f1 = '{f1}',	roc_auc = '{roc_auc}'  where id ='{algo_id}' "
df3 = mysql.run_sql_insert(query_insert2)


# print('mydict')
# print(mydict)


for predictor in predictors:
    print('predictor')
    print(predictor)
    #reset for each predictor
    v_min = None
    v_max = None
    v_avg = None
    v_stdv = None
    
    if(predictor in min_max_data):
        print("min_max_data[predictor]", predictor)
        print(min_max_data[predictor])
        v_min = min_max_data[predictor][0]
        v_max = min_max_data[predictor][1]
        v_avg = min_max_data[predictor][2]
        v_stdv = min_max_data[predictor][3]
        print("v_min")
        print(v_min)
        
    coeff = mydict[predictor]
    query_insert4 = f"insert into stats.ml_coefficients (id,column_name, coefficient, min, max, avg, stdv) values  ('{algo_id}','{predictor}','{coeff}','{v_min}','{v_max}','{v_avg}','{v_stdv}') "
    df4 = mysql.run_sql_insert(query_insert4)
    
   # min_max_data[col] = [min_v_clean, max_v_clean, mean_v_clean, stdv_v_clean]
    
# print('df3')
# print(df3)
#
#
'''

sklearn models

Linear Regression: LinearRegression
Ridge Regression: Ridge
Lasso Regression: Lasso
ElasticNet Regression: ElasticNet
Least Angle Regression: Lars
Orthogonal Matching Pursuit (OMP): OrthogonalMatchingPursuit
Bayesian Ridge Regression: BayesianRidge
Automatic Relevance Determination (ARD): ARDRegression
Logistic Regression: LogisticRegression
Stochastic Gradient Descent (SGD) Classifier and Regressor: SGDClassifier, SGDRegressor
Passive Aggressive Classifier and Regressor: PassiveAggressiveClassifier, PassiveAggressiveRegressor
Perceptron: Perceptron
RANSAC (RANdom SAmple Consensus): RANSACRegressor
Theil-Sen Estimator: TheilSenRegressor
Huber Regression: HuberRegressor
Polynomial Regression


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