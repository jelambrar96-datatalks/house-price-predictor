from .basicmodel import BasicMLModel

import xgboost as xgb

class ZCXGBoost(BasicMLModel):

    def __init__(self, *args, **kwargs):
        self.model = xgb.XGBRegressor(*args, **kwargs)
        self.features = kwargs.get('features', None)
        self.target = kwargs.get('target', None)
        self.params = kwargs.get('params', None)
        self.model = xgb.XGBRegressor(**self.params)
    
    def fit(self, X, y):
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.features)
        self.fitted_model = self.model.fit(dtrain)
    
    def predict(self, X):
        dtest = xgb.DMatrix(X, feature_names=self.features)
        return self.model.fitted_model(dtest)
