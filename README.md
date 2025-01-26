# House Price Predictor

This repository, jelambrar96-datatalks/house-price-predictor, appears to be a machine learning and deep learning project focused on predicting house prices. The repository likely contains code and data for training and testing models to forecast house prices based on various factors.

## 1. Problem Description

The price of a house depends on a wide range of variables, including its location, size, number of bedrooms, age, proximity to amenities, and even market conditions. Each of these factors interacts with others in complex ways. For example, a small house in a prime location may still be priced higher than a larger house in a less desirable area. Accounting for all these variables and their interactions to predict house prices accurately is an extremely challenging task when approached manually.

Trying to manually develop an algorithm to predict house prices by accounting for all these factors and their relationships can quickly become impractical and overwhelming. Manually developing an algorithm to incorporate and weigh these diverse factors would require an exhaustive amount of time, domain expertise, and effort. Not to mention, the relationships between these variables often involve nonlinear patterns, which are difficult to model with traditional rule-based approaches.

This complexity is precisely why machine learning becomes invaluable. Machine learning models are designed to learn from historical data, identifying patterns and relationships between variables without requiring explicit programming for each scenario. By leveraging these models, we can develop a predictive system that efficiently accounts for the wide array of factors influencing house prices, making accurate and scalable predictions possible.

## 2. Proposed solution

This is where machine learning provides an excellent solution. **Machine learning models** are designed to analyze vast datasets, uncover hidden patterns, and identify relationships between variables. By training a model with historical data on house prices and their associated features, we can build a system capable of making accurate predictions for new houses.

Here’s a structured solution to the house price prediction problem:  


### 2.1. **Proposed Solution for Predicting House Prices**  

To address the problem of predicting house prices effectively, we will follow a systematic approach that incorporates data exploration, model development, tracking, and deployment.  


#### **2.1.1. Exploratory Data Analysis (EDA)**  
**Objective**: Understand the dataset and uncover meaningful patterns and relationships.  
- **Data Cleaning**: Handle missing values, remove duplicates, and standardize data formats.  
- **Feature Understanding**: Analyze features such as location, size, number of bedrooms, age, and market conditions.  
- **Univariate Analysis**: Explore individual features to identify distributions, outliers, and trends.  
  - Example: Use histograms for price distribution or bar plots for categorical variables (e.g., number of bedrooms).  
- **Multivariate Analysis**: Analyze the relationships between features.  
  - Example: Correlation matrices and scatter plots to explore how size or location impacts price.  
- **Feature Engineering**:  
  - Create new features, such as price per square foot or proximity to city center.  
  - Encode categorical features (e.g., neighborhoods) using one-hot encoding or target encoding.  

#### **2.1.2. Train Multiple Machine Learning Models**  
**Objective**: Build and evaluate machine learning models to identify the best-performing algorithm.  
- **Data Splitting**: Divide the dataset into training, validation, and testing sets. 
- **Multiple Models**: Train and compare the performance of multiple models:   
- **Model Evaluation**: Use metric Mean Squared Error (MSE) to evaluate performance.  
- **Perform hyperparameter tuning** using techniques like Grid Search or Random Search with cross-validation to improve accuracy.  

#### **2.1.3. Model Registry with MLflow**  
**Objective**: Track experiments and manage models efficiently.  
- **Track Experiments**: Use MLflow to log training parameters, metrics, and artifacts for each model. Log hyperparameters, training/validation scores, and model artifacts (e.g., pickled model files).  
- **Model Registry**: Register the best-performing model in the MLflow Model Registry for easy retrieval. Assign stages such as *Staging*, *Production*, and *Archived* to manage model lifecycles.  

#### **2.1.4. Deploy the Model with Flask**  
**Objective**: Make the model available for real-world use via a REST API.This ensures the model is accessible to end-users via a lightweight and scalable API.  
- **Build API with Flask**:  
  - Create endpoints such as `/predict` to accept house feature data (e.g., size, location, age) as input.  
  - Load the best model from the MLflow registry and use it to generate predictions.  
- **Containerization**: Use Docker to containerize the Flask application for easy deployment.  

### 2.2. Architecture

![archiecture](drawio/archecture.drawio.png)

### 2.3. File Structure

```bash
$ tree . 
.
├── dataset
│   ├── data_description.txt
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── docker-compose.yaml
├── drawio
│   ├── archecture.drawio
│   └── archecture.drawio.png
├── minio_client
│   ├── create_bucket.py
│   └── Dockerfile
├── flask
│   ├── app.py
│   ├── Dockerfile
│   ├── model
│   │   ├── conda.yaml
│   │   ├── MLmodel
│   │   ├── model.pkl
│   │   ├── python_env.yaml
│   │   └── requirements.txt
│   ├── Pipfile
│   ├── Pipfile.lock
│   ├── run.sh
│   ├── templates
│   │   ├── index.html
│   │   └── result.html
│   └── test
│       ├── test_reload.py
│       └── test_requests.py
├── mlflow
│   ├── Dockerfile
│   ├── requirements.txt
│   └── run.sh
├── mlzoomcamp_evaluation_criteria
│   └── 3113934462-waffle_k_ltr.css
├── mlzoomcamp_evaluation_criteria.html
├── notebooks
│   ├── eda.ipynb
│   └── requirements.txt
├── README.md
├── sample.env
└── train
    ├── docker-compose.yaml
    ├── Dockerfile
    ├── get_best_model.ipynb
    ├── main.py
    ├── Pipfile
    ├── Pipfile.lock
    ├── run.sh
    ├── sample.env
    ├── sample.json
    ├── train.csv
    ├── train.ipynb
    ├── train_notebook.ipynb
    └── train_pipelines_sklearn.ipynb
```

## 3. Exploratory Data Analysis (EDA)

Click to go to [notebook](notebooks/eda.ipynb)

This notebook performs an Exploratory Data Analysis (EDA) on the House Prices - Advanced Regression Techniques dataset. The notebook provides a comprehensive initial analysis of the dataset, cleaning and transforming the data to prepare it for further modeling.

The analysis includes:

1. **Introduction**: Overview of the dataset's structure and objectives of the EDA.
2. **Imports**: Importing necessary libraries: `numpy`, `pandas`, `matplotlib.pyplot`, and `seaborn`.
3. **Data Loading**: Loading the training dataset from a CSV file.
4. **Column Analysis**:
   - Renaming columns to snake_case.
   - Displaying the first five rows of the dataset.
   - Providing a summary of the data columns.
5. **Null Values Analysis**: Identifying columns with null values and calculating the percentage of null values.
6. **Data Cleaning**: Dropping less useful columns and rows with missing values.
7. **Target Column Analysis**:
   - Visualizing the distribution of the target variable (`sale_price`).
   - Transforming `sale_price` using `np.log1p` to normalize the distribution.
8. **Categorical and Numerical Data**:
   - Identifying categorical and numerical columns.
   - Displaying the count of each categorical column.
9. **Visualizing Categorical Columns**:
   - Generating bar plots for each categorical column to visualize the distribution of categories.
10. **Feature importance Analysis**: Apply a DecisionTreeRegressor to identify most relevant variables. 

## 4. Training models

The training **Jupyter notebook** performs the following steps:

1. **Setup and Imports**:
   - Imports necessary libraries such as `numpy`, `pandas`, `sklearn`, and others.
   - Sets environment variables for accessing `MLFLOW` and `MINIO`.
   
2. **Data Loading and Preparation**:
   - Downloads the training dataset.
   - Reads the dataset into a DataFrame and preprocesses column names.
   - Drops columns with significant null values and rows with any null values.
   - Transforms the target column (`sale_price`) using `np.log1p`.
   - Splits the dataset into training and testing sets.
   - Scales numerical features and vectorizes categorical features.

3. **Model Training and Hyperparameter Search**:
   - Utilizes different models from `sklearn` with hyperparameter tuning using `GridSearchCV` or `RandomizedSearchCV`.

The following `sklearn` models are used along with their hyperparameters:

1. **Linear Regression**:
   - Hyperparameters: `fit_intercept` (True, False)
   - Search Method: `GridSearchCV`

2. **Lasso Regression**:
   - Hyperparameters: `alpha` (0.001, 0.01, 0.1, 1, 10, 100)
   - Search Method: `GridSearchCV`

3. **Decision Tree Regressor**:
   - Hyperparameters: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `random_state`
   - Search Method: `RandomizedSearchCV`

4. **Random Forest Regressor**:
   - Hyperparameters: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`, `oob_score`, `criterion`, `ccp_alpha`, `random_state`
   - Search Method: `RandomizedSearchCV`

5. **AdaBoost Regressor**:
   - Hyperparameters: `learning_rate`, `n_estimators`, `loss`, `estimator`, `random_state`
   - Search Method: `RandomizedSearchCV`

6. **Gradient Boosting Regressor**:
   - Hyperparameters: `learning_rate`, `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `subsample`, `loss`, `alpha`, `random_state`
   - Search Method: `RandomizedSearchCV`

7. **MLP Regressor** (Neural Network):
   - Hyperparameters: `hidden_layer_sizes`, `activation`, `solver`, `learning_rate_init`, `batch_size`, `learning_rate`, `power_t`, `max_iter`, `early_stopping`, `validation_fraction`
   - Search Method: `RandomizedSearchCV`

Each model's best estimator, parameters, and score are determined using the respective search method. You can view the full notebook [here](train/train_pipelines_sklearn.ipynb).

### 4.1. Separated python script

All training process can be execute using [main.py](train/main.py) file. 


## 5. Deployment with `Flask`

The [app.py](flask/app.py) script sets up a Flask web application to serve machine learning predictions. The application exposes API endpoints for making predictions and managing the model, and it provides a web interface for user interaction. Here's a breakdown of its components and functionality:

1. **Imports and Libraries**: The script imports various libraries needed for the application, including Flask for the web server, boto3 for AWS interactions, numpy and pandas for data handling, sklearn for machine learning, and mlflow for model tracking.

2. **Environment Variables**: It loads configuration settings from environment variables (e.g., AWS credentials, MLflow server details).

3. **Base JSON Structure**: Defines a template JSON (`BASE_JSON`) with default values for various house attributes. This is used to ensure all necessary fields are present when preparing data for the model.

4. **Function `prepare_features`**: This function takes input data and fills in any missing values using the defaults from `BASE_JSON`.

5. **MLflow Experiment Setup**: If an MLflow server is specified, it sets the tracking URI and loads the experiment for tracking model runs.

6. **Function `load_sklearn_model`**: Downloads and loads a scikit-learn model artifact from MLflow and returns the loaded model.

7. **Temporary Directory**: Creates a temporary directory for storing the model locally.

8. **Class `ModelLoader`**: 
   - Manages loading the machine learning model.
   - Provides methods to load a model from MLflow or from a local file.
   - Checks if the model is ready and performs predictions.

9. **Flask Application Setup**: 
   - Initializes the Flask app.
   - Defines routes for reloading the model (`/api/reload`), making predictions (`/api/predict`), and rendering the home page (`/` and `/index`).

   - **Endpoint `/api/reload`**: Reloads the model and returns success or failure.
   - **Endpoint `/api/predict`**: Accepts JSON input, prepares features, and returns predictions.
   - **Endpoint `/index`**: Renders a web form for user input, handles form submissions, and displays predictions.

10. **Run the Flask App**: Starts the Flask server on `0.0.0.0` at port `8000` in debug mode.

In summary, this script sets up a Flask web application that serves predictions from a machine learning model. It supports interaction with MLflow for model management and uses MINIO for potential cloud storage integration. 

## 6. Dependencies Management

Managing dependencies is crucial for ensuring that your project runs smoothly in different environments. Below is the dependency management strategy for each module in the repository:

### 6.1. Train Module
The `train` module uses `Pipenv` for dependency management. The dependencies are specified in the `Pipfile` and `Pipfile.lock`.

- **Pipfile**: Contains the project’s dependencies and their versions.
- **Pipfile.lock**: Ensures reproducible builds by specifying exact versions of dependencies.

To install the dependencies for the `train` module, run the following command:
```bash
pipenv install
```

### 6.2. MLflow Module
The `mlflow` module uses a `requirements.txt` file to manage dependencies.

- **requirements.txt**: Lists the dependencies required for the MLflow module.

To install the dependencies for the `mlflow` module, run the following command:
```bash
pip install -r requirements.txt
```

### 6.3. Flask Module
The `flask` module also uses `Pipenv` for dependency management. The dependencies are specified in the `Pipfile` and `Pipfile.lock`.

To install the dependencies for the `flask` module, run the following command:
```bash
pipenv install
```

### 6.4. MinIO Client Module
The `minio_client` module manages its dependencies through a `Dockerfile`.

- **Dockerfile**: Specifies the necessary dependencies and environment setup for the MinIO client module.

To build the Docker image for the `minio_client` module, run the following command:
```bash
docker build -t minio_client .
```

### 6.5. Notebooks for EDA
The `notebooks` directory uses a `requirements.txt` file to manage dependencies for Exploratory Data Analysis (EDA).

To install the dependencies for the EDA notebooks, run the following command:
```bash
pip install -r notebooks/requirements.txt
```


## 7. Docker and containerization

Containerization is essential for making sure that the application runs consistently across different environments. Below is the containerization strategy for each module in the repository:

### 7.1. Train Module
The `train` module is containerized using a `Dockerfile`. The `docker-compose.yaml` can be used to orchestrate multiple services if needed.

- **Dockerfile**: Specifies the environment and dependencies for the training module.
- **docker-compose.yaml**: Orchestrates the services for the training module.

To build and run the Docker container for the `train` module, use the following commands:
```bash
docker build -t train_module .
docker run train_module
```

### 7.2. MLflow Module
The `mlflow` module is also containerized using a `Dockerfile`.

- **Dockerfile**: Contains the necessary steps to set up the MLflow environment and install dependencies.

To build and run the Docker container for the `mlflow` module, use the following commands:
```bash
docker build -t mlflow_module .
docker run mlflow_module
```

### 7.3. Flask Module
The `flask` module is containerized using a `Dockerfile`.

- **Dockerfile**: Specifies the environment and dependencies for the Flask application.

To build and run the Docker container for the `flask` module, use the following commands:
```bash
docker build -t flask_module .
docker run -p 8000:8000 flask_module
```

### 7.4. MinIO Client Module
The `minio_client` module uses a `Dockerfile` for containerization.

- **Dockerfile**: Contains the steps to set up the MinIO client environment.

To build and run the Docker container for the `minio_client` module, use the following commands:
```bash
docker build -t minio_client .
docker run minio_client
```

## 8. Reproducitibility

To run the House Price Predictor project, follow these steps:

1. **Create an Environment File**:
   Copy the sample environment file to create your own `.env` file.
   ```bash
   cp sample.env .env
   ```

2. **Modify Your .env File**:
   Open the `.env` file and modify the environment variables according to your configuration. Ensure that you set values for `MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`, `JUPYTER_TOKEN`, and other necessary variables.

3. **Start the Docker Containers**:
   Use `docker-compose` to bring up the services defined in the `docker-compose.yaml` file. This will start all the required containers in detached mode.
   ```bash
   docker-compose --env-file .env up -d
   ```


___________________


[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/jelambrar1)

Made with Love ❤️ by [@jelambrar96](https://github.com/jelambrar96)
