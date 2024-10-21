import joblib
import boto3_rds_pandas as mysql
algo_id =  '1715007944_Logistic_Regression_fit_transform.joblib'

#from EC2
#get sklearn object as job_object from local storage on ec2
try:
    job_object = joblib.load(f'{algo_id}.joblib')
    job_object_dict= (job_object.__dict__)
except Exception as e:
    print('failed to load ', algo_id)  
    print('error:  ', e)  
#unpacked_job_object_keys = dir(job_object)

# from RDS    
#algo_data from database
try:
    query = f"SELECT * FROM stats.ml_algorithms where id = '{algo_id}';"
    algo_data = mysql.run_sql_query(query)
except Exception as e:
    print('failed to find algo_id: ', algo_id," in stats.ml_algorithms")  
    print('error:  ', e) 
    
#unpack algo_data
job_object_data = {}
for key in algo_data:
    job_object_data[key] = algo_data[key][0]

#combine both dicts
job_object_data.update(job_object_dict)
print('job_object_data:  ', job_object_data) 

