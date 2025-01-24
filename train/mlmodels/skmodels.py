from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from .basicmodel import BasicModel


class ZCLinearRegression(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = LinearRegression(*args, **kwargs)
        self.fitted_model = None

    def fit(self, X, y):
        # print("fit")
        self.fitted_model = self.model.fit(X, y)

    def predict(self, X):
        # print("predict")
        return self.fitted_model.predict(X)

    def get_params(self):
        return self.fitted_model.get_params()


class ZCLassoRegression(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = Lasso(*args, **kwargs)
        self.fitted_model = None

    def fit(self, X, y):
        # print("fit")
        self.fitted_model = self.model.fit(X, y)

    def predict(self, X):
        # print("predict")
        return self.fitted_model.predict(X)

    def get_params(self):
        return self.fitted_model.get_params()


class ZCDecisionTreeRegressor(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = DecisionTreeRegressor(*args, **kwargs)
        self.fitted_model = None

    def fit(self, X, y):
        # print("fit")
        self.fitted_model = self.model.fit(X, y)

    def predict(self, X):
        # print("predict")
        return self.fitted_model.predict(X)

    def get_params(self):
        return self.fitted_model.get_params()


class ZCRandomForestRegressor(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = RandomForestRegressor(*args, **kwargs)
        self.fitted_model = None

    def fit(self, X, y):
        # print("fit")
        self.fitted_model = self.model.fit(X, y)

    def predict(self, X):
        # print("predict")
        return self.fitted_model.predict(X)

    def get_params(self):
        return self.fitted_model.get_params()    


class ZCAdaBoostRegressor(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = AdaBoostRegressor(*args, **kwargs)
        self.fitted_model = None

    def fit(self, X, y):
        # print("fit")
        self.fitted_model = self.model.fit(X, y)

    def predict(self, X):
        # print("predict")
        return self.fitted_model.predict(X)

    def get_params(self):
        return self.fitted_model.get_params()


class ZCGradientBoostingRegressor(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = GradientBoostingRegressor(*args, **kwargs)
        self.fitted_model = None

    def fit(self, X, y):
        # print("fit")
        self.fitted_model = self.model.fit(X, y)

    def predict(self, X):
        # print("predict")
        return self.fitted_model.predict(X)

    def get_params(self):
        return self.fitted_model.get_params()
