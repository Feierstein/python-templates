
import boto3
import pymysql
import pandas as pd
import numpy as np
# Assuming you've already configured your AWS credentials using AWS CLI or IAM roles
show_logs = False # True
#show_logs = True 

# Create a boto3 RDS client
rds_client = boto3.client('rds')

# Describe RDS instances
response = rds_client.describe_db_instances()
if(show_logs == True): print(response)

# Print instance information
if(show_logs == True):
    for db_instance in response['DBInstances']:
        print('Instance identifier:', db_instance['DBInstanceIdentifier'])
        print('Instance status:', db_instance['DBInstanceStatus'])
        print('Endpoint:', db_instance['Endpoint']['Address'])
        print('Port:', db_instance['Endpoint']['Port'])
        print('Engine:', db_instance['Engine'])
        print('Engine version:', db_instance['EngineVersion'])
        print('---------------------------------------')
    

def run_sql_query(query):
    # Connect to the RDS instance
    rds = boto3.client('rds')
    db_instance_identifier = 'fcapidev-replica'
    db_instance = rds.describe_db_instances(DBInstanceIdentifier=db_instance_identifier)['DBInstances'][0]
    endpoint = db_instance['Endpoint']['Address']
    port = db_instance['Endpoint']['Port']
    username = 'cloud9'
    password = 'TLgCM56^BZ'
    db_name = 'bitrail'

    # Connect to the database
    conn = pymysql.connect(host=endpoint, port=port, user=username, password=password, database=db_name)

    try:
        # Execute the SQL query
        with conn.cursor() as cursor:
            cursor.execute(query)
            result = cursor.fetchall()
            columns = [col[0] for col in cursor.description] # Get column names
            data_frame = pd.DataFrame(result, columns=columns)
            clean_data_frame = clean_data(data_frame)
            #print('clean_data_frame')
            #print(clean_data_frame)
            return clean_data_frame
            
            
            #return pd.DataFrame(result, columns=columns) # Convert result to DataFrame
    finally:
        # Close the connection
        conn.close()
        

query = "SELECT * FROM stats.user_scores_both limit 1"
# result1 = run_sql_query(query)
# print('results1')
# print(result1)

def run_sql_insert(query):
    # Connect to the RDS instance
    rds = boto3.client('rds')
    db_instance_identifier = 'fcapidev-replica'
    db_instance = rds.describe_db_instances(DBInstanceIdentifier=db_instance_identifier)['DBInstances'][0]
    endpoint = db_instance['Endpoint']['Address']
    port = db_instance['Endpoint']['Port']
    username = 'cloud9'
    password = 'TLgCM56^BZ'
    db_name = 'bitrail'

    # Connect to the database
    conn = pymysql.connect(host=endpoint, port=port, user=username, password=password, database=db_name)

    # Create a cursor object
    cur = conn.cursor()
    
    print('query')
    print(query)
    # Execute the insert statement
    cur.execute(query)
    #cur.execute(sql, (data['id'], data['name'], data['type'], data['y_col']))
    # Commit the transaction
    conn.commit()
    
    # Close cursor and connection
    cur.close()
    conn.close()
            
    
    #query = "SELECT * FROM stats.user_scores_both limit 1"
    # result1 = run_sql_query(query)
    # print('results1')
    # print(result1)



def clean_data(data):
    #data = data.replace({None: np.nan})
    data = data.replace({None: 0})
    return data