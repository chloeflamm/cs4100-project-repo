import numpy as np

class FFNN:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

    # Initializing activation functions
    def relu(self, z): # hidden layer operation
        return np.maximum(0, z)

    def relu_deriv(self, z): # backpropogation operation returning 1 (if positive) or 0 (if negative)
        return (z > 0).astype(float)

    def softmax(self, z): # output operation, converts sums to probabilities that sum to 1
        z = z - z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def compute_loss(self, y_pred, y_true):
        n = y_true.shape[0]
        log_probs = -np.log(y_pred[np.arange(n), y_true] + 1e-8)
        return np.mean(log_probs)
