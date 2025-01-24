import os
import re

from datetime import datetime

import numpy as np
import pandas as pd


from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import root_mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor



import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature



RANDOM_SEED = 42


AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY", None); 
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY", None); 

MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", None); 
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", None); 
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", None); 
MLFLOW_S3_ENDPOINT_URL = os.getenv("MLFLOW_S3_ENDPOINT_URL", None); 
MLFLOW_S3_IGNORE_TLS = os.getenv("MLFLOW_S3_IGNORE_TLS", None); 
MLFLOW_BUCKET_NAME = os.getenv("MLFLOW_BUCKET_NAME", None); 
MLFLOW_SERVER = os.getenv("MLFLOW_SERVER", None);
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "mlzoomcamp");


# Function to convert camelCase or PascalCase to snake_case
def to_snake_case(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


TARGET_COLUMN = "sale_price"


df_full = pd.read_csv("train.csv")
df_full.columns = [to_snake_case(col) for col in df_full.columns]

df_full.drop(
    columns=["id", "alley", "pool_qc", "fence", "misc_feature", "mas_vnr_type", "fireplace_qu", "lot_frontage"],
    inplace=True
)
df_full.dropna(inplace=True)
df_full[TARGET_COLUMN] = np.log1p(df_full[TARGET_COLUMN])


# -----------------------------------------------------------------------------

df_full_train, df_test = train_test_split(
    df_full, test_size=0.2, random_state=RANDOM_SEED)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = (df_full_train[TARGET_COLUMN]).astype('int').values
y_test = (df_test[TARGET_COLUMN]).astype('int').values

del df_full_train[TARGET_COLUMN]
del df_test[TARGET_COLUMN]

# -----------------------------------------------------------------------------

categorical_cols = df_full_train.select_dtypes(include=['object']).columns
dv = DictVectorizer(sparse=False)

full_train_dict = df_full_train.to_dict(orient='records')
X_full_train = dv.fit_transform(full_train_dict)

test_dict = df_test.to_dict(orient='records')
X_test = dv.transform(test_dict)

# -----------------------------------------------------------------------------

SCORING = "neg_root_mean_squared_error"
mlflow.set_tracking_uri(MLFLOW_SERVER)

list_experiments = mlflow.search_experiments(
    filter_string=f"name = '{MLFLOW_EXPERIMENT_NAME}'")
if len(list_experiments) == 0:
    mlflow.create_experiment(
        MLFLOW_EXPERIMENT_NAME,
        artifact_location=f"s3://{MLFLOW_BUCKET_NAME}/experiments/") 
    list_experiments = mlflow.search_experiments(
        filter_string=f"name = '{MLFLOW_EXPERIMENT_NAME}'")

mlflow_experiment = list_experiments[0]
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# print(mlflow_experiment)
# print(mlflow_experiment.experiment_id)

# print(MLFLOW_TRACKING_URI)

def print_model_version_info(mv):
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    print(f"Source: {mv.source}")

client = mlflow.MlflowClient()

# -----------------------------------------------------------------------------

linear_regression_params = {
    'dict_vectorizer__sparse': [False],
    'linear_regression__fit_intercept': [True, False]
}

linear_regession_pipeline = Pipeline([
    ('dict_vectorizer', DictVectorizer()),
    ('linear_regression', LinearRegression())
])


linear_regression_grid_search = GridSearchCV(
    estimator=linear_regession_pipeline,
    param_grid=linear_regression_params,
    n_jobs=-1,
    scoring=SCORING
)
linear_regression_grid_search_fitted = linear_regression_grid_search.fit(full_train_dict, y_full_train)

best_linear_regression_estimator = linear_regression_grid_search_fitted.best_estimator_
best_linear_regression_params = linear_regression_grid_search_fitted.best_params_
best_linear_regression_score = linear_regression_grid_search_fitted.best_score_


# model 1 lienar regression
with mlflow.start_run(
    experiment_id=mlflow_experiment.experiment_id,
    run_name="linear-regression") as run:
    #
    y_test_pred = best_linear_regression_estimator.predict(test_dict)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    signature = infer_signature(test_dict, y_test_pred)    
    # 
    mlflow.log_metric("rmse", rmse)
    mlflow.log_params(best_linear_regression_params)
    mlflow.log_metric(f'{SCORING}', best_linear_regression_score)
    mlflow.sklearn.log_model(best_linear_regression_estimator, "model", signature=signature)

    src_name = f'linear-regression-staging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    client.create_registered_model(src_name)
    src_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
    
    # Copy the source model version into a new registered model
    dst_name = "linear-regression-production"
    src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
    mv_copy = client.copy_model_version(src_model_uri, dst_name)
    print_model_version_info(mv_copy)



lasso_params = {
    'dict_vectorizer__sparse': [False],
    'lasso_regression__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
}

lasso_regession_pipeline = Pipeline([
    ('dict_vectorizer', DictVectorizer()),
    ('lasso_regression', Lasso())
])

lasso_grid_search = GridSearchCV(
    estimator=lasso_regession_pipeline,
    param_grid=lasso_params,
    n_jobs=-1,
    scoring=SCORING
)
lasso_grid_search_fitted = lasso_grid_search.fit(full_train_dict, y_full_train)

best_lasso_estimator = lasso_grid_search_fitted.best_estimator_
best_lasso_params = lasso_grid_search_fitted.best_params_
best_lasso_score = lasso_grid_search_fitted.best_score_


# model 1 lienar regression
with mlflow.start_run(
    experiment_id=mlflow_experiment.experiment_id,
    run_name="lasso-regression") as run:
    #
    y_test_pred = best_lasso_estimator.predict(test_dict)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    signature = infer_signature(test_dict, y_test_pred)    
    # 
    mlflow.log_metric("rmse", rmse)
    mlflow.log_params(best_lasso_params)
    mlflow.log_metric(f'{SCORING}', best_lasso_score)
    mlflow.sklearn.log_model(best_lasso_estimator, "model", signature=signature)

    src_name = f'lasso-regression-staging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    client.create_registered_model(src_name)
    src_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
    
    # Copy the source model version into a new registered model
    dst_name = "lasso-regression-production"
    src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
    mv_copy = client.copy_model_version(src_model_uri, dst_name)
    print_model_version_info(mv_copy)

# -----------------------------------------------------------------------------

decision_tree_params = {
    'dict_vectorizer__sparse': [False],
    'decision_tree_regressor__criterion': ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    'decision_tree_regressor__max_depth': [None, 5, 10, 20],
    'decision_tree_regressor__min_samples_split': [2, 5, 10],
    'decision_tree_regressor__min_samples_leaf': [1, 5, 10],
    'decision_tree_regressor__max_features': ["sqrt", "log2", 0.5, 1.0],
    'decision_tree_regressor__random_state': [RANDOM_SEED]
}

decision_tree_pipeline = Pipeline([
    ('dict_vectorizer', DictVectorizer()),
    ('decision_tree_regressor', DecisionTreeRegressor())
])

decision_tree_search = RandomizedSearchCV(
    estimator=decision_tree_pipeline,
    param_distributions=decision_tree_params,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    scoring=SCORING
)

decision_tree_search_fitted = decision_tree_search.fit(full_train_dict, y_full_train)

best_decision_tree_estimator = decision_tree_search_fitted.best_estimator_
best_decision_tree_params = decision_tree_search_fitted.best_params_
best_decision_tree_score = decision_tree_search_fitted.best_score_


with mlflow.start_run(
    experiment_id=mlflow_experiment.experiment_id,
    run_name="decision-tree-regression") as run:
    #
    y_test_pred = best_decision_tree_estimator.predict(test_dict)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    signature = infer_signature(test_dict, y_test_pred)    
    # 
    mlflow.log_metric("rmse", rmse)
    mlflow.log_params(best_decision_tree_params)
    mlflow.log_metric(f'{SCORING}', best_decision_tree_score)
    mlflow.sklearn.log_model(best_decision_tree_estimator, "model", signature=signature)

    src_name = f'decision-tree-regression-staging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    client.create_registered_model(src_name)
    src_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
    
    # Copy the source model version into a new registered model
    dst_name = "decision-tree-regression-production"
    src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
    mv_copy = client.copy_model_version(src_model_uri, dst_name)
    print_model_version_info(mv_copy)


# ------------------------------------------------------------------------------


random_forest_params = {
    "dict_vectorizer__sparse": [False],
    "random_forest_regressor__n_estimators": [10, 50, 100, 200],
    "random_forest_regressor__max_depth": [None, 5, 10, 20],
    "random_forest_regressor__min_samples_split": [2, 5, 10],
    "random_forest_regressor__min_samples_leaf": [1, 5, 10],
    "random_forest_regressor__max_features": ["sqrt", "log2", 0.5, 1.0],
    "random_forest_regressor__bootstrap": [True, False],
    "random_forest_regressor__oob_score": [True, False],
    "random_forest_regressor__criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
    "random_forest_regressor__ccp_alpha": [0.0, 0.1, 0.2],
    "random_forest_regressor__random_state": [RANDOM_SEED]
}

random_forest_pipeline = Pipeline([
    ('dict_vectorizer', DictVectorizer()),
    ('random_forest_regressor', RandomForestRegressor())
])

random_forest_search = RandomizedSearchCV(
    estimator=random_forest_pipeline,
    param_distributions=random_forest_params,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    scoring=SCORING
)
random_forest_search_fitted = random_forest_search.fit(full_train_dict, y_full_train)

best_random_forest_estimator = random_forest_search_fitted.best_estimator_
best_random_forest_params = random_forest_search_fitted.best_params_
best_random_forest_score = random_forest_search_fitted.best_score_


with mlflow.start_run(
    experiment_id=mlflow_experiment.experiment_id,
    run_name="random-forest-regression") as run:
    #
    y_test_pred = best_random_forest_estimator.predict(test_dict)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    signature = infer_signature(test_dict, y_test_pred)    
    # 
    mlflow.log_metric("rmse", rmse)
    mlflow.log_params(best_random_forest_params)
    mlflow.log_metric(f'{SCORING}', best_random_forest_score)
    mlflow.sklearn.log_model(best_random_forest_estimator, "model", signature=signature)

    src_name = f'random-forest-staging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    client.create_registered_model(src_name)
    src_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
    
    # Copy the source model version into a new registered model
    dst_name = "random-forest-regression-production"
    src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
    mv_copy = client.copy_model_version(src_model_uri, dst_name)
    print_model_version_info(mv_copy)


# -----------------------------------------------------------------------------

adaboost_regressor_params = {
    "dict_vectorizer__sparse": [False],
    "adaboost_regressor__learning_rate": [0.01, 0.1, 1],
    "adaboost_regressor__n_estimators": [10, 50, 100, 200],
    "adaboost_regressor__loss": ["linear", "square", "exponential"],
    "adaboost_regressor__estimator": [
        DecisionTreeRegressor(),
        RandomForestRegressor(),
        # DecisionTreeRegressor(**best_decision_tree_params),
        # RandomForestRegressor(**best_random_forest_params)
    ],
    "adaboost_regressor__random_state": [RANDOM_SEED]
}

adaboost_pipeline = Pipeline([
    ('dict_vectorizer', DictVectorizer()),
    ('adaboost_regressor', AdaBoostRegressor())
])

adaboost_regressor_search = RandomizedSearchCV(
    estimator=adaboost_pipeline,
    param_distributions=adaboost_regressor_params,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    scoring=SCORING
)
adaboost_regressor_search_fitted = adaboost_regressor_search.fit(full_train_dict, y_full_train)

best_adaboost_regressor_estimator = adaboost_regressor_search_fitted.best_estimator_
best_adaboost_regressor_params = adaboost_regressor_search_fitted.best_params_
best_adaboost_regressor_score = adaboost_regressor_search_fitted.best_score_

# model 1
with mlflow.start_run(
    experiment_id=mlflow_experiment.experiment_id,
    run_name="adaboost-regression") as run:
    #
    y_test_pred = best_adaboost_regressor_estimator.predict(test_dict)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    signature = infer_signature(test_dict, y_test_pred)    
    # 
    mlflow.log_metric("rmse", rmse)
    mlflow.log_params(best_adaboost_regressor_params)
    mlflow.log_metric(f'{SCORING}', best_adaboost_regressor_score)
    mlflow.sklearn.log_model(best_adaboost_regressor_estimator, "model", signature=signature)

    src_name = f'adaboost-regression-staging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    client.create_registered_model(src_name)
    src_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
    
    # Copy the source model version into a new registered model
    dst_name = "adaboost-regression-production"
    src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
    mv_copy = client.copy_model_version(src_model_uri, dst_name)
    print_model_version_info(mv_copy)


# -----------------------------------------------------------------------------

gradientboost_regressor_params = {
    "dict_vectorizer__sparse": [False],
    "gradientboost_regressor__learning_rate": [0.1, 0.05, 0.01],
    "gradientboost_regressor__n_estimators": [50, 100, 200],
    "gradientboost_regressor__max_depth": [3, 5, 7],
    "gradientboost_regressor__min_samples_split": [2, 5, 10],
    "gradientboost_regressor__min_samples_leaf": [1, 2, 4],
    "gradientboost_regressor__max_features": ["auto", "sqrt", "log2"],
    "gradientboost_regressor__subsample": [1.0, 0.8, 0.5],
    "gradientboost_regressor__loss": ["squared_error", "absolute_error", "huber"],
    "gradientboost_regressor__alpha": [0.5, 0.75, 0.9],
    "gradientboost_regressor__random_state": [RANDOM_SEED]
}

grandientboost_regressor_pipeline = Pipeline([
    ('dict_vectorizer', DictVectorizer()),
    ('gradientboost_regressor', GradientBoostingRegressor())
])

gradientboost_regressor_search = RandomizedSearchCV(
    estimator=grandientboost_regressor_pipeline,
    param_distributions=gradientboost_regressor_params,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    scoring=SCORING
)
gradientboost_regressor_search_fitted = gradientboost_regressor_search.fit(full_train_dict, y_full_train)


best_gradientboost_regressor_estimator = gradientboost_regressor_search_fitted.best_estimator_
best_gradientboost_regressor_params = gradientboost_regressor_search_fitted.best_params_
best_gradientboost_regressor_score = gradientboost_regressor_search_fitted.best_score_


# model
with mlflow.start_run(
    experiment_id=mlflow_experiment.experiment_id,
    run_name="gradientboost-regression") as run:
    #
    y_test_pred = best_gradientboost_regressor_estimator.predict(test_dict)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    signature = infer_signature(test_dict, y_test_pred)    
    # 
    mlflow.log_metric("rmse", rmse)
    mlflow.log_params(best_gradientboost_regressor_params)
    mlflow.log_metric(f'{SCORING}', best_gradientboost_regressor_score)
    mlflow.sklearn.log_model(best_gradientboost_regressor_estimator, "model", signature=signature)

    src_name = f'gradientboost-regression-staging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    client.create_registered_model(src_name)
    src_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
    
    # Copy the source model version into a new registered model
    dst_name = "gradientboost-regression-production"
    src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
    mv_copy = client.copy_model_version(src_model_uri, dst_name)
    print_model_version_info(mv_copy)


# ------------------------------------------------------------------------

mlp_regressor_params = {
    'dict_vectorizer__sparse': [False],
    'mlp_regressor__hidden_layer_sizes': [(32,), (64,), (128,)],
    'mlp_regressor__activation': ['relu', 'tanh', 'logistic'],
    'mlp_regressor__solver': ['lbfgs', 'adam'],
    'mlp_regressor__learning_rate_init': [0.001, 0.01, 0.1],
    'mlp_regressor__batch_size': ['auto', 200],
    'mlp_regressor__learning_rate': ['constant', 'invscaling', 'adaptive'],
    'mlp_regressor__power_t': [0.5, 1.0, 2.0],
    'mlp_regressor__max_iter': [500, 1000, 1500],
    'mlp_regressor__early_stopping': [True, False],
    'mlp_regressor__validation_fraction': [0.2, 0.5, 0.8]
}

mlp_regressor_pipeline = Pipeline([
    ('dict_vectorizer', DictVectorizer()),
    ('mlp_regressor', MLPRegressor())
])

mlp_regressor_search = RandomizedSearchCV(
    estimator=mlp_regressor_pipeline,
    param_distributions=mlp_regressor_params,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    scoring=SCORING
)
mlp_regressor_search_fitted = mlp_regressor_search.fit(full_train_dict, y_full_train)

best_mlp_regressor_estimator = mlp_regressor_search_fitted.best_estimator_
best_mlp_regressor_params = mlp_regressor_search_fitted.best_params_
best_mlp_regressor_score = mlp_regressor_search_fitted.best_score_


with mlflow.start_run(
    experiment_id=mlflow_experiment.experiment_id,
    run_name="mlp-regression") as run:
    #
    y_test_pred = best_mlp_regressor_estimator.predict(test_dict)
    rmse = root_mean_squared_error(y_test, y_test_pred)
    signature = infer_signature(test_dict, y_test_pred)    
    # 
    mlflow.log_metric("rmse", rmse)
    mlflow.log_params(best_mlp_regressor_params)
    mlflow.log_metric(f'{SCORING}', best_mlp_regressor_score)
    mlflow.sklearn.log_model(best_mlp_regressor_estimator, "model", signature=signature)

    src_name = f'mlp-regression-staging-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    client.create_registered_model(src_name)
    src_uri = f"runs:/{run.info.run_id}/sklearn-model"
    mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
    
    # Copy the source model version into a new registered model
    dst_name = "mlp-regression-production"
    src_model_uri = f"models:/{mv_src.name}/{mv_src.version}"
    mv_copy = client.copy_model_version(src_model_uri, dst_name)
    print_model_version_info(mv_copy)

