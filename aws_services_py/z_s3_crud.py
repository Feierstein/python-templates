import boto3

# create an s3 bucket
s3 = boto3.resource('s3')
bucket_name  = 'feierstein-boto3-bucket'

all_my_buckets = [bucket.name for bucket in s3.buckets.all()]
if bucket_name not in all_my_buckets:
    print(f"'{bucket_name}' bucket does not exist.  creating now")
    s3.create_bucket(Bucket = bucket_name)
else: 
    print(f"'{bucket_name}' bucket already exists.")
    
    
    