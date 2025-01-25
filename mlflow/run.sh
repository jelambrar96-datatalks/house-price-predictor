docker build -t mlflow .
docker run --rm mlflow mlflow server --host 0.0.0.0 --port 5000