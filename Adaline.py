import numpy.random
import numpy as np

class Adaline:
    def __init__(self, learning_rate=0.01, epochs=200, batch=50, weights = None, random_state=None, 
                 tol=1e-4, early_stopping=False, solver = None):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch = batch
        self.random_state = numpy.random(random_state)
        self.weights = weights
        self.bias = None
        self.tol = tol
        self.early_stopping = early_stopping
            
    def net_input(self, X):
        #Takes the input features
        return np.dot(self.weights[1:], X) + self.bias

    def activation(self, X):
        #Returns the net_input function value
        return self.net_input(X)
    
    def _gd(self, X, y):
        # Gradient Descent
        """" The general formula for gradient descent is used in this function"""
        try:
            x_new = X - self.learning_rate * np.gradient(X) 
        except IndexError:
            error = y - self.activation(X)
            self.weights += self.learning_rate * x_new.T.dot(error)
            self.bias += self.learning_rate * error.mean()
        
        return self.weights, self.bias
    
    def _sgd(self, X, y):
        # Stochastic Gradient Descent
        """ The general formula for stochastic gradient descent is used in this function """
        
        if self.weights is None:
            self.weights = np.zeros(X.shape[1] + 1)
        
        if self.bias is None:
            self.bias = 0.0

        for i in range(len(X)):
            xi = X[i:i+1]
            yi = y[i:i+1]
            try:
                self.weights -= self.learning_rate * np.gradient(self.weights[xi, yi])
                self.bias -= self.learning_rate * np.gradient(self.bias[xi, yi])
            except IndexError:        
                error = yi - self.activation(xi)
                self.weights += self.learning_rate * xi.T.dot(error)
                self.bias += self.learning_rate * error.mean()
        
        return self.weights, self.bias

    def fit(self, X, y):
        self.cost = []
        best_cost = np.inf
        rounds_without_improvement = 0

        for epoch in range(self.epochs):
            error = y - self.activation(X)
            self.weights += self.learning_rate * X.T.dot(error)
            self.bias += self.learning * error.mean()
            cost = np.sum(error ** 2) / 2

            #Early stopping logic
            if self.early_stopping is not False:
                if cost < best_cost - self.tol:
                    best_cost = cost
                    rounds_without_improvement = 0
                else:
                    rounds_without_improvement += 1
                if rounds_without_improvement >= self.early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
            
            if self.solver is not None:
                if self.solver == 'gd':
                    return self._gd(X, y)
                elif self.solver == 'sgd':
                    return self._sgd(X, y)
                else:
                    raise ValueError("Unknown solver type. Use 'gd' or 'sgd'.")
                self._sgd(X, y)
            
            self.cost.append(cost)

        return self.cost

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1.0, -1)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
model = Adaline(learning_rate=0.01, epochs=1000, tol=1e-3, early_stopping_rounds=10)
