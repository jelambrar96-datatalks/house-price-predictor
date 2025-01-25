"""
Flask application for serving machine learning predictions.

This application loads a machine learning model and serves predictions via a REST API.
It interacts with S3 for model storage and uses LocalStack for local development.
The model predicts outcomes based on ride data, such as pickup and dropoff locations.

"""

import os
import pickle
import tempfile

import boto3
import pandas as pd
import sklearn  # pylint: disable=unused-import

import mlflow  # pylint: disable=unused-import
from flask import Flask, jsonify, request

from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.model_utils import (
    _add_code_from_conf_to_system_path,
    _get_flavor_configuration,
)

# -----------------------------------------------------------------------------
# ENV VARIABLES
# -----------------------------------------------------------------------------

# Load AWS credentials and S3 bucket information from environment variables

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", None); 
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", None); 
AWS_REGION = os.getenv("AWS_REGION", None)

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", None); 
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", None); 

MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", None);
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None); 
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", None); 

MLFLOW_S3_IGNORE_TLS = os.getenv("MLFLOW_S3_IGNORE_TLS", None); 
MLFLOW_BUCKET_NAME = os.getenv("MLFLOW_BUCKET_NAME", None); 
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlzoomcamp");


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
#
# -----------------------------------------------------------------------------

# Load the MLflow experiment
mlflow.set_tracking_uri(MLFLOW_SERVER)

experiment = mlflow.search_experiments(filter_string=f"name='{MLFLOW_EXPERIMENT_NAME}'")[0]
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# -----------------------------------------------------------------------------


def load_sklearn_model(model_uri, dst_path):
    local_model_path = _download_artifact_from_uri(artifact_uri=model_uri, output_path=dst_path)
    # print(local_model_path)
    # shutil.copytree(f'{local_model_path}/model', f'{local_model_path}/MLmodel')
    local_model_path = os.path.join(local_model_path, 'MLmodel')
    flavor_conf = _get_flavor_configuration(
        model_path=local_model_path,
        flavor_name="sklearn"
    )
    _add_code_from_conf_to_system_path(local_model_path, flavor_conf)
    sklearn_model_artifacts_path = os.path.join(local_model_path, flavor_conf["pickled_model"])
    serialization_format = flavor_conf.get("serialization_format", "pickle")
    with open(sklearn_model_artifacts_path, 'rb') as f:
        loaded_model = pickle.load(f)
    return loaded_model


# -----------------------------------------------------------------------------
# Create a temporary directory for storing the model and vectorizer locally
# -----------------------------------------------------------------------------

TEMP_MODEL_DIR = "/tmp/model/skmodel"
os.makedirs(TEMP_MODEL_DIR, exist_ok=True)

# -----------------------------------------------------------------------------

class ModelLoader:


    def __init__(self):
        self._model = None
    
    def load_model(self):
        runs_df = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if len(runs_df) == 0:
            return False
        try:
            best_run = runs_df.sort_values(by="metrics.rmse", ascending=True).iloc[0]
            model_path = best_run.artifact_uri
            self._model = load_sklearn_model(model_uri=model_path, dst_path='model')
            return True
        except:
            return False
    
    def load_model_from_file(self, filename):
        with open(filename, 'rb') as f:
            self._model = pickle.load(f)

    def is_ready(self):
        return not self._model is None

    def predict(self, X):
        return self._model.predict(X)



# Instantiate the ModelLoader class
model_loader = ModelLoader()
model_loader.load_model_from_file("model/model.pkl")
# Instantiate the Flask app
app = Flask(__name__)


@app.route('/api/reload', methods=['POST'])
def reload_endpoint():
    """
    Reload the model and vectorizer via an API call.

    Returns:
        Response: JSON response indicating success or failure of the reload operation.
    """
    flag = model_loader.load_model()
    if flag:
        return jsonify({"result": "success", "reloaded": True})
    return jsonify({"result": "failed", "reloaded": False})


@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    """
    Predict outcomes using the loaded model via an API call.

    Returns:
        Response: JSON response containing the predictions.
    """
    ride = request.get_json()["data"]
    features = prepare_features(ride)
    pred = model_loader.predict(features)
    result = "failed"
    if pred is not None:
        result = "success"
    return jsonify({"result": result, "predictions": pred})


@app.route('/', methods=['GET'])
def index():
    """
    Root endpoint to verify the service is running.

    Returns:
        str: Welcome message indicating the service is running.
    """
    return "Zoomcamp application"


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8000)
