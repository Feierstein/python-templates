import boto3

s3 = boto3.resource('s3')

for bucket in s3.buckets.all():
    print(bucket.name)

#touch boto3_setup.py creates this file    
#pip install boto3