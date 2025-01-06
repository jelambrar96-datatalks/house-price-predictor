
class BasicModel:

    def __init__(self, *args, **kwargs):
        self.model = None

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        self.model.predict(X)


