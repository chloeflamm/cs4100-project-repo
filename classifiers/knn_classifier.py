"""
knn_classifier.py
Author: Nasir Stanley

K-Nearest Neighbors classifier implemented from scratch using numpy. Predicts T-cell activation state by majority vote
among k nearest neighbors in PCA-reduced gene expression space.
"""
import numpy as np

class KNN:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self.class_to_idx_ = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        # Store classes for probability vector construction
        self.classes_ = np.unique(y)
        self.class_to_idx_ = {c: i for i, c in enumerate(self.classes_)}
    '''
    AI Usage: How can i get my predictions to run faster than an iterative loop?
    '''
    def predict(self, X, batch_size=500):
        """
        Batched vectorized prediction with vote fraction probabilities for estimations.
        Computes distances for 500 cells at once instead of one by one.
        Returns (predictions, probabilities).
        """
        predictions  = []
        probabilities = []
        n = len(X)

        for i in range(0, n, batch_size):
            batch = X[i:i + batch_size]

            # Euclidean distance from each cell in batch to every training cell
            # Shape: (batch_size, n_train)
            dists = np.sqrt(
                ((batch[:, np.newaxis, :] - self.X_train[np.newaxis, :, :]) ** 2).sum(axis=2))

            # Indices of knn for each cell
            k_indices = np.argsort(dists, axis=1)[:, :self.k]

            for indices in k_indices:
                k_labels = self.y_train[indices]
                values, counts = np.unique(k_labels, return_counts=True)

                # Majority vote of predicted class
                predictions.append(values[np.argmax(counts)])

                # Vote fractions as probability estimate over all classes
                prob = np.zeros(len(self.classes_))
                for label, count in zip(values, counts):
                    prob[self.class_to_idx_[label]] = count / self.k
                probabilities.append(prob)

            print(f"Predicted {min(i + batch_size, n)}/{n} cells", end="\r")

        print()
        return np.array(predictions), np.array(probabilities)