
import sys
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import time
import boto3_rds_pandas as mysql
import joblib
import sklearn_preprocess as pp
import sklearn_models as sm

print("running sklearn_create_algo.py")

def user_risk_score_decision_tree():
    print("running sklearn_create_algo.user_risk_score_decision_tree")
    algo_type = 'Decision Tree'
    algo_id =  str(int(time.time())) + '_' + algo_type
    algo_name =  'Decision Tree user_risk'
    algo_object = {}
    y_col = "resulted_in_reconcilation" ## known values of whether or not a transaction resulted in a reconciliation
    show_logs = False
    query = "SELECT transaction_id, months, transaction_type,  trust_status, complete_sales_a, reconciliations_c,user_has_ssn,resulted_in_reconcilation FROM stats.user_scores_both WHERE 1=1  "
    algo_object.update({'algo_type': algo_type,'algo_id': algo_id, 'algo_name': algo_name,'y_col': y_col, 'query': query})
    
    df = mysql.run_sql_query(query)
    data = df.copy() # Keep a copy our original data 
    #preprocessed_data = pp.preprocess(df, algo_type,algo_id, algo_name, y_col)   
    preprocessed_data = pp.preprocess(df, algo_object)
    x_train = preprocessed_data[0]
    x_test = preprocessed_data[1]
    y_train = preprocessed_data[2]
    y_test = preprocessed_data[3]
    objects_converted = preprocessed_data[4]
    #query = query.lower()
    sm.decision_tree( algo_object, x_train, x_test, y_train, y_test,objects_converted)
    # query_insert = f"insert into stats.ml_algorithms (created_at,id,name,type,y_col,sql_used) values(now(),'{algo_id}','{clf}','{y_col}','{query}')"
    # mysql.run_sql_insert(query_insert)


def user_risk_score_logistical():
    print("running sklearn_create_algo.user_risk_score_logistical")
    algo_object = {}
    algo_type = 'Logistic_Regression'
    algo_id =  str(int(time.time())) + '_' + algo_type
    algo_name =  'logistical_user_risk'
    y_col = "resulted_in_reconcilation" ## known values of whether or not a transaction resulted in a reconciliation
    show_logs = False
    query = "SELECT transaction_id, months, emails, phone_numbers, bank_accounts, complete_ach_c, complete_orders_c, transaction_type, kyc_method, trust_status, complete_sales_a, reconciliations_c, user_has_ssn, resulted_in_reconcilation FROM stats.user_scores_both WHERE 1=1  "
    algo_object.update({'algo_type': algo_type,'algo_id': algo_id, 'algo_name': algo_name,'y_col': y_col, 'query': query})
    
    df = mysql.run_sql_query(query)
    data_bu = df.copy() # Keep a copy our original data 
    
    #clean y out before sending predictors
    df_x = df.drop(y_col, axis=1)

    # run data through established preprocessing model
    preprocessed_data = pp.preprocess(df, algo_object)
    x_train = preprocessed_data[0]
    x_test = preprocessed_data[1]
    y_train = preprocessed_data[2]
    y_test = preprocessed_data[3]
    objects_converted = preprocessed_data[4]
    
    print('objects_converted',objects_converted)
    
    sm.logistical(algo_object, x_train, x_test, y_train, y_test, objects_converted, df_x)

def user_score_elastic_net():
    print("running sklearn_create_algo.user_score_elastic_net")
    algo_object = {}
    algo_type = 'Elastic Net'
    algo_id =  str(int(time.time())) + '_' + algo_type
    algo_name =  'Elastic Net user_risk'
    y_col = "resulted_in_reconcilation" ## known values of whether or not a transaction resulted in a reconciliation
    show_logs = False
    query = "SELECT transaction_id, emails, phone_numbers, bank_accounts, complete_ach_c, complete_orders_c, transaction_type, kyc_method, trust_status, complete_sales_a, reconciliations_c, user_has_ssn, resulted_in_reconcilation FROM stats.user_scores_both WHERE 1=1  "
    algo_object.update({'algo_type': algo_type,'algo_id': algo_id, 'algo_name': algo_name,'y_col': y_col, 'query': query})
    
    df = mysql.run_sql_query(query)
    data_bu = df.copy() # Keep a copy our original data 
    
    # run data through established preprocessing model
    preprocessed_data = pp.preprocess(df, algo_object)
    x_train = preprocessed_data[0]
    x_test = preprocessed_data[1]
    y_train = preprocessed_data[2]
    y_test = preprocessed_data[3]
    
    objects_converted = preprocessed_data[4]
    #object_type =  type(objects_converted)
    print('objects_converted1', objects_converted)
    objects_converted = objects_converted.replace("'", "")
    print('objects_converted2', objects_converted)

    sm.elastic_net(algo_object, x_train, x_test, y_train, y_test,objects_converted)
    
    
    
def user_risk_score_logistical_v2():
    print("running sklearn_create_algo.user_risk_score_logistical_v2")
    algo_object = {}
    algo_type = 'Logistic_Regression'
    algo_id =  str(int(time.time())) + '_' + algo_type
    algo_name =  'logistical_user_risk'
    y_col = "resulted_in_reconcilation" ## known values of whether or not a transaction resulted in a reconciliation
    show_logs = False
    query = "SELECT transaction_id, months, emails, phone_numbers, bank_accounts, complete_ach_c, complete_orders_c, transaction_type, kyc_method, trust_status, complete_sales_a, reconciliations_c, user_has_ssn, resulted_in_reconcilation FROM stats.user_scores_both WHERE 1=1  "
    algo_object.update({'algo_type': algo_type,'algo_id': algo_id, 'algo_name': algo_name,'y_col': y_col, 'query': query})
    
    df = mysql.run_sql_query(query)
    data_bu = df.copy() # Keep a copy our original data 
    
    
    
    #exploratory data analysis
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Summary statistics
    print(df.describe())
    
    # Visualizations
    sns.pairplot(df)
    plt.show()
    #exploratory data analysis
    
    
    
    #clean y out before sending predictors
    df_x = df.drop(y_col, axis=1)

    # run data through established preprocessing model
    preprocessed_data = pp.preprocess(df, algo_object)
    x_train = preprocessed_data[0]
    x_test = preprocessed_data[1]
    y_train = preprocessed_data[2]
    y_test = preprocessed_data[3]
    objects_converted = preprocessed_data[4]
    
    print('objects_converted',objects_converted)
    
    sm.logistical(algo_object, x_train, x_test, y_train, y_test, objects_converted, df_x)
    
if __name__ == "__main__":
    user_risk_score_logistical()
