import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            dists = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
            k_indices = np.argsort(dists)[:self.k]
            k_labels = self.y_train[k_indices]
            values, counts = np.unique(k_labels, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        return np.array(predictions)

