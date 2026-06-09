import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, batch_size=500):
        predictions = []
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            dists = np.sqrt(((batch[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2).sum(axis=2))
            k_indices = np.argsort(dists, axis=1)[:, :self.k]
            for indices in k_indices:
                k_labels = self.y_train[indices]
                values, counts = np.unique(k_labels, return_counts=True)
                predictions.append(values[np.argmax(counts)])
            print(f"  Predicted {min(i+batch_size, len(X))}/{len(X)} cells", end="\r")
        return np.array(predictions)