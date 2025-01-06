from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor

from .basicmodel import BasicModel


class ZCLinearRegression(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = LinearRegression(*args, **kwargs)


class ZCLassoRegression(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = Lasso(*args, **kwargs)


class ZCDesitionTreeRegressor(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = DecisionTreeRegressor(*args, **kwargs)


class ZCRandomForestRegressor(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = RandomForestRegressor(*args, **kwargs)
    

class ZCAdaBoostRegressor(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = AdaBoostRegressor(*args, **kwargs)


class ZCGradientBoostingRegressor(BasicModel):
    
    def __init__(self, *args, **kwargs):
        self.model = GradientBoostingRegressor(*args, **kwargs)
