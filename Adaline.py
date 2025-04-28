import numpy.random
class Adaline:
    def __init__(self, learning_rate=0.01, epochs=200, batch=True, random_state=X_train, weights=np.zeros(X.shape[1] + 1)):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch = batch
        self.random_state = numpy.random(random_state)
        self.weights = weights
        self.bias = None
            
    def net_input(self, X):
        #Takes the input features
        return np.dot(self.weights[1:], X) + self.bias

    def activation(self, X):
        #Returns the net_input function value
        return self.net_input(X)

    def fit(self, X, y):
        self.cost = []

        for epoch in range(self.epochs):
            ...
            error = y - self.activation(X)
            self.weights += self.learning_rate * X.T.dot(error)
            self.bias += self.learning * error.mean()
            if not True:
                for i in range(self.random_state):
                    error = y[i] - self.activation(X[i])
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                errors = np.sum(error ** 2) / 2

            self.cost.append(errors)
        return self.cost

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1.0, -1)