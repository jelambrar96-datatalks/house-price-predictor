# FROM python:3.10-slim
FROM ubuntu:22.04

RUN mkdir -p /home/mlflow
RUN chmod 777 /home/mlflow

WORKDIR /home/mlflow

COPY requirements.txt ./
RUN apt-get update
RUN apt-get install -y python3.10 python3-pip
RUN pip install --no-cache-dir -r requirements.txt


ENV MLFLOW_DEFAULT_ARTIFACT_ROOT=$MLFLOW_DEFAULT_ARTIFACT_ROOT
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV MLFLOW_BACKEND_STORE_URI=$MLFLOW_BACKEND_STORE_URI
ENV MLFLOW_ARTIFACT_STORE=$MLFLOW_ARTIFACT_STORE
ENV MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL
ENV POSTGRES_USER=$POSTGRES_USER
ENV POSTGRES_PASSWORD=$POSTGRES_PASSWORD
ENV POSTGRES_DB=$POSTGRES_DB

EXPOSE 5000

# CMD [ \
#     "mlflow", "server", \
#     "--backend-store-uri", "'$MLFLOW_BACKEND_STORE_URI'", \
#     "--default-artifact-root", "'$MLFLOW_DEFAULT_ARTIFACT_ROOT'", \
#     "--host", "0.0.0.0", \
#     "--port", "5000" \
# ]

# CMD ["sh", "-c", "mlflow server \
#     --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
#     --default-artifact-root $MLFLOW_DEFAULT_ARTIFACT_ROOT \
#     --host 0.0.0.0 --port 5000"]

# CMD ["sh", "-c", "mlflow server \
#     --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
#     --port 5000 \
#     --host 0.0.0.0 \
#     --serve-artifacts \
#     --artifacts-destination ${MLFLOW_DEFAULT_ARTIFACT_ROOT}"]

