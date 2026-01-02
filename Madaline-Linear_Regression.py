import numpy as np
import matplotlib.pyplot as plt

class HybridLRAdaline:
    def __init__(self, learning_rate=0.01, n_iterations=100, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.weights = None
        self.cost_history = []
    
    def fit(self, X, y, use_refinement=True):
        X_with_bias = np.c[np.ones(X.shape[0]), X]
        m, n = X_with_bias.shape
        
        print("Phase 1: Linear Regression Initialisation")
        self.weights = self._linear_regression_closed_form(X_with_bias, y)
        print(f"Initial weights: {self.weights}")