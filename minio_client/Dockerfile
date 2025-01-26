# Use an official lightweight Python image
FROM python:3.10-bookworm

# Set the working directory in the container
WORKDIR /app

# Copy the script into the container
COPY create_bucket.py .

# Install required Python packages
RUN pip install boto3 minio

# Define the entry point for the container
ENTRYPOINT ["python", "create_bucket.py"]
