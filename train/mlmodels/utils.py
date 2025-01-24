import pickle
import minio
import requests

def save_model(model, path):
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path):
    model = None
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def create_bucket_minio(minio_client, bucket_name):
    minio_client.make_bucket(bucket_name)

def load_file_minio(minio_client, file_path, butcket_name, object_path):
    minio_client.fput_object(bucket_name, object_path, file_path)

def download_csv_file(url, file_path):
    res = requests.get(url)
    with open(file_path, 'w') as f:
        f.write(res.text)
