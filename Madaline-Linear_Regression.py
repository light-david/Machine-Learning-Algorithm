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

        initial_cost = self._compute_cost(X_with_bias, y)
        self.cost_history.append(initial_cost)
        print(f"Initial cost: {initial_cost:.6f}")

        if use_refinement:
            print(f"\nPhase 2: Adaline Refinement ({self.n_iterations} iterations)")
            self._adaline_refinement(X_with_bias, y, m)
            print(f"Final weights: {self.weights}")
            print(f"Final cost: {self.cost_history[-1]:.6f}")
        return self
    
    def _linear_regression_closed_form(self, X, y):
        """
        Compute weights using closed-form solution: w = (X^T X)^-1 X^T y
        """
        return np.linalg.inv(X.T @ X) @ X.T @ y
    
    def _adaline_refinement(self, X, y, m):
        """
        Refine weights using Adaline (bacth gradeint descent)
        """
        for i in range(self.n_iterations):
            y_pred = X @ self.weights
            errors = y - y_pred

            gradient = (self.learning_rate / m) * (X.T @ errors)
            w_old = self.weights.copy()
            self.weights += gradient

            cost = self._compute_cost(X, y)
            self.cost_history.append(cost)

            weight_change = np.linalg.norm(self.weights - w_old)
            if weight_change < self.tolerance:
                print(f"Converged at iteartion {i+1}")
                break

            if (i + 1) % 10 == 0:
                print(f"Iteration {i+1}: Cost = {cost:.6f}, Weight change = {weight_change:.6f}")

    def _compute_cost(self, X, y):
        """
        Compute SUm of Squared Errors cost function: J(w) = 1/2 * sum((y - y_pred)^2)
        """
        y_pred = X @ self.weights
        return 0.5 * np.sum((y - y_pred) ** 2)
    
    def predict(self, X):
        """
        Parameters:
        -------------------
        X : array-like, shape (m, n)
            input data
        
        Returns:
        -------------------
        predicitions : array, shape (m,)
        """
        X_with_bias = np.c_[np.ones(X,shape[0]), X]
        return X_with_bias @ self.weights
    
    def plot_cost_history(self):
        """
        Plot the cost function ever iterations
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history, 'b-', linewidth=2)
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabael('Cost (SSE)', fontsize=12)
        plt.title('Cost Function Over Iterations', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt,show()

"""
Sample Usage
"""
if __name__ == "__main__":
    np.random.seed(42)
    m = 100
    X = 2 * np.random.rand(m, 1)
    y = 4 + 3 * X. flatten() + np.random.randn(m)

    print("=" * 60)
    print("Hybrid Linear Regression-Adaline Example")
    print("=" * 60)
    print(f"Training samples: {m}")
    print(f"True parameters: intercept=4, slope=3")
    print("=" * 60)
    
    #Train with refinement
    model = HybridLRAdaline(learning_rate=0.01, n_iterations=100, tolerance=1e-6)
    model.fit(X, y, use_refinement=True)

    #Make predictions
    X_test = np.array([[0], [2]])
    predictions = model.predict(X_test)
    print(f"\nPredictions:")
    print(f"X=0: y={predictions[0]:.4f}")
    print(f"X=2: y={predictions[1]:.4f}")

    #Plot results
    plt.figure(figsize=(12, 5))

    #Plot 1: Data and fitted line
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='Training data')
    X_line = np.linespace(0, 2, 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='Fitted line')
    plt.xlabel('X', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('hybrid LR-Adaline Fit', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)

    #Plot 2: Cost history
    plt.subplot(1, 2, 2)
    plt.plot(range(len(model.cost_history)), model.cost_history, 'b-', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Cost (SSE)', fontsize=12)
    plt.title('Cost Function Over Iterations', fontsize=14)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print("\n" + "=" * 60)
    print("Training complete!👍")
    print("=" * 60)