
class BasicModel:

    def __init__(self, *args, **kwargs):
        self.model = None
        self.fitted_model = None

    def fit(self, X, y):
        # print("fit")
        self.fitted_model = self.model.fit(X, y)

    def predict(self, X):
        # print("predict")
        return self.fitted_model.predict(X)

