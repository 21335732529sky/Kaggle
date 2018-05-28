from sklearn.tree import DecisionTreeClassifier

class Model:
    def __init__(self):
        self.model = DecisionTreeClassifier()

    def train(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)
