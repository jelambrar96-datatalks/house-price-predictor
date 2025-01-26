import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def create_minio_bucket():
    # Fetch credentials and server details from environment variables
    aws_access_key = os.getenv('AWS_ACCESS_KEY')
    aws_secret_key = os.getenv('AWS_SECRET_KEY')
    minio_server = os.getenv('MINIO_SERVER')
    bucket_name = os.getenv('BUCKET_NAME', 'default-bucket')  # Default bucket name

    if not aws_access_key or not aws_secret_key or not minio_server:
        print("Error: Missing required environment variables.")
        return

    try:
        # Initialize S3 client for MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url=minio_server,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key
        )

        # Create the bucket
        if not bucket_name:
            print("Error: No bucket name provided.")
            return

        # Check if the bucket already exists
        existing_buckets = s3_client.list_buckets()
        if any(bucket['Name'] == bucket_name for bucket in existing_buckets['Buckets']):
            print(f"Bucket '{bucket_name}' already exists.")
        else:
            s3_client.create_bucket(Bucket=bucket_name)
            print(f"Bucket '{bucket_name}' created successfully.")

    except NoCredentialsError:
        print("Error: Invalid credentials.")
    except PartialCredentialsError:
        print("Error: Partial credentials provided.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    create_minio_bucket()
