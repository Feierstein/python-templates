import boto3
import pandas as pd
from io import BytesIO

s3 = boto3.resource('s3')
#static                #bucket name                        #file name
# obj = s3.Object('sagemaker-data-feierstein','user_score_both_all_fields_20230516.csv')
# body = obj.get()['Body'].read
# print(body)
bucket_name = 'sagemaker-data-feierstein'
file_name = 'user_score_both_all_fields_20230516.csv'
obj = s3.Object(bucket_name, file_name)
body = obj.get()['Body'].read()
df = pd.read_csv(BytesIO(body))
print(df)

# bucket_name = 'sagemaker-data-feierstein'
# file_name = 'user_score_both_all_fields_20230516.csv'
# obj = s3.Object(bucket_name, file_name)
# body = obj.get()['Body'].read()
# df = pd.read_csv(BytesIO(body))
# print(df)