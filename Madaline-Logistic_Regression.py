import numpy as np
import matplotlib.pyplot as plt

class MADALINELogistic:
    """
    Hybrid model combining MADALINE architecture with Logistic Regression.
    Uses sigmoid activations throughout for probabilistic outputs.
    """
    
    def __init__(self, n_features, n_hidden_units=3, learning_rate=0.1, random_state=42):
        """
        Initialize the MADALINE-Logistic model.
        
        Parameters:
        -----------
        n_features : int
            Number of input features
        n_hidden_units : int
            Number of ADALINE units in hidden layer
        learning_rate : float
            Learning rate for gradient descent
        random_state : int
            Random seed for reproducibility
        """
        np.random.seed(random_state)
        
        self.n_features = n_features
        self.n_hidden_units = n_hidden_units
        self.lr = learning_rate
        
        # Initialize weights for hidden layer (ADALINE units)
        # Each row represents weights for one ADALINE unit
        self.W_hidden = np.random.randn(n_hidden_units, n_features) * 0.5
        self.b_hidden = np.zeros(n_hidden_units)
        
        # Initialize weights for output layer (Logistic Regression)
        self.W_output = np.random.randn(n_hidden_units) * 0.5
        self.b_output = 0.0
        
        self.loss_history = []
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))  # Clip to prevent overflow
    
    def forward(self, X):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Input data
            
        Returns:
        --------
        h : Hidden layer activations
        y_pred : Output predictions (probabilities)
        """
        # Hidden layer: Multiple ADALINE units with sigmoid activation
        z_hidden = np.dot(X, self.W_hidden.T) + self.b_hidden  # (n_samples, n_hidden_units)
        h = self.sigmoid(z_hidden)
        
        # Output layer: Logistic regression
        z_output = np.dot(h, self.W_output) + self.b_output  # (n_samples,)
        y_pred = self.sigmoid(z_output)
        
        return h, y_pred
    
    def compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def backward(self, X, y_true, h, y_pred):
        """
        Backward pass using backpropagation.
        
        Parameters:
        -----------
        X : Input data
        y_true : True labels
        h : Hidden layer activations
        y_pred : Predicted probabilities
        """
        m = X.shape[0]
        
        # Output layer gradients
        dz_output = y_pred - y_true  # (n_samples,)
        dW_output = np.dot(h.T, dz_output) / m  # (n_hidden_units,)
        db_output = np.mean(dz_output)
        
        # Hidden layer gradients
        dh = np.outer(dz_output, self.W_output)  # (n_samples, n_hidden_units)
        dz_hidden = dh * h * (1 - h)  # Sigmoid derivative
        dW_hidden = np.dot(dz_hidden.T, X) / m  # (n_hidden_units, n_features)
        db_hidden = np.mean(dz_hidden, axis=0)  # (n_hidden_units,)
        
        # Update weights
        self.W_output -= self.lr * dW_output
        self.b_output -= self.lr * db_output
        self.W_hidden -= self.lr * dW_hidden
        self.b_hidden -= self.lr * db_hidden
    
    def fit(self, X, y, epochs=1000, verbose=True):
        """
        Train the model using gradient descent.
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels (0 or 1)
        epochs : int
            Number of training iterations
        verbose : bool
            Whether to print progress
        """
        X = np.array(X)
        y = np.array(y)
        
        for epoch in range(epochs):
            # Forward pass
            h, y_pred = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)
            
            # Backward pass
            self.backward(X, y, h, y_pred)
            
            if verbose and (epoch + 1) % 100 == 0:
                accuracy = np.mean((y_pred >= 0.5).astype(int) == y)
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        X = np.array(X)
        _, y_pred = self.forward(X)
        return y_pred
    
    def predict(self, X, threshold=0.5):
        """Predict class labels"""
        return (self.predict_proba(X) >= threshold).astype(int)


# Example usage with XOR problem (non-linearly separable)
if __name__ == "__main__":
    print("=" * 60)
    print("MADALINE-Logistic Regression Hybrid Model")
    print("=" * 60)
    
    # XOR dataset (classic non-linear problem)
    X_xor = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_xor = np.array([0, 1, 1, 0])
    
    print("\nTraining on XOR problem...")
    print("Input data:")
    print(X_xor)
    print("\nTarget labels:")
    print(y_xor)
    print()
    
    # Create and train model
    model = MADALINELogistic(n_features=2, n_hidden_units=4, learning_rate=0.5)
    model.fit(X_xor, y_xor, epochs=2000, verbose=True)
    
    # Test predictions
    print("\n" + "=" * 60)
    print("Final Predictions:")
    print("=" * 60)
    predictions = model.predict(X_xor)
    probabilities = model.predict_proba(X_xor)
    
    for i, (x, y_true, y_pred, prob) in enumerate(zip(X_xor, y_xor, predictions, probabilities)):
        print(f"Input: {x} | True: {y_true} | Predicted: {y_pred} | Probability: {prob:.4f}")
    
    accuracy = np.mean(predictions == y_xor)
    print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")
    
    # Plot learning curve
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(model.loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.grid(True, alpha=0.3)
    
    # Visualize decision boundary
    plt.subplot(1, 2, 2)
    h = 0.01
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.8)
    plt.colorbar(label='Probability')
    plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1], 
                c='blue', s=100, edgecolors='k', label='Class 0')
    plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1], 
                c='red', s=100, edgecolors='k', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("Model Architecture:")
    print("=" * 60)
    print(f"Input Layer: {model.n_features} features")
    print(f"Hidden Layer: {model.n_hidden_units} ADALINE units (sigmoid)")
    print(f"Output Layer: 1 logistic unit (sigmoid)")
    print(f"Total Parameters: {model.W_hidden.size + model.b_hidden.size + model.W_output.size + 1}")