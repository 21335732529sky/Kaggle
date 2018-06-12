from sklearn.ensemble import GradientBoostingClassifier

class Model:
    def __init__(self, params):
        self.model = GradientBoostingClassifier(**params)

    def train(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)
