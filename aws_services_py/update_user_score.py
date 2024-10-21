import sys
# Surpress warnings:
def warn(*args, **kwargs):
    pass
import warnings
import json
import boto3_rds_pandas as mysql
import sklearn_use_algorithm as use_algo
import time


def update_user(user_id, algo_id, algo_result, type_):
    sql = f"INSERT INTO stats.user_score (updated_at, user_id, algo_id, algo_result, type) VALUES (now(),'{user_id}','{algo_id}', '{algo_result}', '{type_}') ON DUPLICATE KEY UPDATE updated_at = now(), algo_id = '{algo_id}', algo_result = '{algo_result}', type = '{type_}' "

    try:
        df = mysql.run_sql_insert(sql)
        print('user:', user_id,'updated')
        #print(df)
    except Exception as e:
        print('user:', user_id,' not updated')
    
    
def update_all_users():
    start_time = time.time()
    result_object = use_algo.run_algo_on_all()
    algo_id = result_object.pop('algo_id')
    type_ = result_object.pop('type_')
    print('result_object',result_object)

    for user_id in result_object:
        update_user(user_id, algo_id, result_object[user_id], type_)
    elapsed_time = time.time() - start_time
    print("Elapsed time in seconds:", elapsed_time)


def update_all_users_bulk():
    start_time = time.time()
    result_object = use_algo.run_algo_on_all()
    algo_id = result_object.pop('algo_id')
    type_ = result_object.pop('type_')
    print('result_object',result_object)
    
    #format object into array
    values_formatted = ''
    i=0
    for user in result_object:
        if i!=0:
            values_formatted = values_formatted + ' , '
        i+=1
        #print('user', user)
        #print('result_object[user]',result_object[user])
        row = f"('{user}',{result_object[user]},'{algo_id}','{type_}', now())"
        print('row', row)
        values_formatted = values_formatted + row
        
        

    sql = f"INSERT INTO stats.user_score (user_id,algo_result, algo_id, type, updated_at) VALUES ({values_formatted}) ON DUPLICATE KEY UPDATE user_id = VALUES(user_id), algo_result = VALUES(algo_result), algo_id = VALUES(algo_id), type = VALUES(type), updated_at = VALUES(updated_at)";
    print('sql', sql)
    inserted_sql = mysql.run_sql_insert(sql)
    print('inserted_sql', inserted_sql)
    #for user_id in result_object:
    #    update_user(user_id, algo_id, result_object[user_id], type_)
    elapsed_time = time.time() - start_time
    print("Elapsed time in seconds:", elapsed_time)
    return 'update_all_users_bulk completed'
    
    
#toggle function for testing
if __name__ == "__main__":
    update_all_users_bulk()
    
