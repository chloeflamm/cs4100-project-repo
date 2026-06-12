"""
Original structure/code written by Jasdeep Singh, modified by Chloe Flamm after Jasdeep left the project.
This file implements a random forest classifier from scratch.
Chloe's contributions: added functionality to return prob estimates for the AUC-ROC evaluation, 
made some minor code cleanup edits, and added comments to help my understanding.
"""

import numpy as np


# a node is either a question (split) or a final answer (leaf)
class Node:
    def __init__(self):
        self.feature = None # which column to split on
        self.threshold = None # the value to compare against
        self.left = None # samples <= threshold go here
        self.right = None # samples > threshold go here
        self.label = None # only set if this is a leaf


# decision tree, RF will make a lot of these when running
class DecisionTree:
    def __init__(self, max_depth=20, n_features=None, max_thresholds=10):
        self.max_depth = max_depth
        self.n_features = n_features
        # max_thresholds limits threshold candidates per feature for speed
        # otherwise would check every unique value, which is slow on large data
        self.max_thresholds = max_thresholds
        self.root = None

    def fit(self, X, y):
        """Builds tree recursively starting from the root node."""
        self.root = self.build_tree(X, y, depth=0)

    def predict(self, X):
        """Route each sample through the tree to get a predicted label."""
        return np.array([self.traverse(sample, self.root) for sample in X])

    def build_tree(self, X, y, depth):
        """Recursively builds the tree by finding the best split at each node."""
        num_samples = X.shape[0]
        num_classes = len(np.unique(y))

        # stop if max depth reached, node is pure, or too few samples to split
        if depth >= self.max_depth or num_classes == 1 or num_samples < 2:
            leaf = Node()
            leaf.label = self.get_majority_class(y)
            return leaf

        # random feature subset at each split — key to RF diversity
        all_features    = list(range(X.shape[1]))
        chosen_features = np.random.choice(all_features, size=self.n_features, replace=False)

        best_feature, best_threshold = self.find_best_split(X, y, chosen_features)

        # if no valid split found, return a leaf
        if best_feature is None:
            leaf = Node()
            leaf.label = self.get_majority_class(y)
            return leaf

        left_idx  = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] >  best_threshold

        node = Node()
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self.build_tree(X[left_idx],  y[left_idx],  depth + 1)
        node.right = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return node

    def find_best_split(self, X, y, features_to_try):
        """Finds best feature/threshold by minimizing weighted gini impurity."""
        best_gini = float('inf')
        best_feature = None
        best_threshold = None

        for feature in features_to_try:
            thresholds = np.unique(X[:, feature])
            # randomly sample thresholds if too many candidates
            if len(thresholds) > self.max_thresholds:
                thresholds = np.random.choice(thresholds, size=self.max_thresholds, replace=False)

            for threshold in thresholds:
                left_y  = y[X[:, feature] <= threshold]
                right_y = y[X[:, feature] >  threshold]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = self.weighted_gini(left_y, right_y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def gini(self, y):
        """
        Gini impurity — measures class mixing in a node.
        0 = pure (all one class), higher = more mixed.
        Lower is better for T-cell classification splits.
        """
        if len(y) == 0:
            return 0
        score = 1.0
        for c in np.unique(y):
            p = np.sum(y == c) / len(y)
            score -= p ** 2
        return score

    def weighted_gini(self, left_y, right_y):
        """Weighted average of child impurities, weighted by child size."""
        total = len(left_y) + len(right_y)
        return (len(left_y) / total) * self.gini(left_y) + \
               (len(right_y) / total) * self.gini(right_y)

    def traverse(self, sample, node):
        """Return label at leaf, otherwise keep traversing."""
        if node.label is not None:
            return node.label
        if sample[node.feature] <= node.threshold:
            return self.traverse(sample, node.left)
        return self.traverse(sample, node.right)

    def get_majority_class(self, y):
        """Returns the most frequent class label in y."""
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

    def predict_probs(self, X, classes):
        """
        Returns a one-hot probability vector per sample instead of a label.
        Single trees are deterministic so probs are always 0 or 1.
        Returns (n_samples x n_classes) matrix.
        """
        return np.array([self.traverse_probs(sample, self.root, classes) for sample in X])

    def traverse_probs(self, sample, node, classes):
        """
        Same as traverse() but returns prob vector at each leaf:
        1 for leaf's class, 0 for all others.
        """
        if node.label is not None:
            prob = np.zeros(len(classes))
            idx = np.where(classes == node.label)[0]
            if len(idx) > 0:
                prob[idx[0]] = 1
            return prob
        if sample[node.feature] <= node.threshold:
            return self.traverse_probs(sample, node.left, classes)
        return self.traverse_probs(sample, node.right, classes)


# RF: just a bunch of trees voting together
class RandomForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        # sqrt(n_features) is the standard subset size per split
        n_features_per_split = int(np.sqrt(X.shape[1]))

        for i in range(self.n_trees):
            # bootstrap: random sample with replacement introduces tree diversity
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[idx]
            y_sample = y[idx]

            tree = DecisionTree(max_depth=self.max_depth, n_features=n_features_per_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

            if (i + 1) % 10 == 0:
                print(f"  Built {i+1}/{self.n_trees} trees", end="\r")
        print()

    def predict(self, X):
        """
        Majority vote across all trees. Returns predicted class labels for each sample.
        """
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        # ensure 2D shape
        if all_preds.ndim == 1:
            all_preds = all_preds.reshape(1, -1)

        final = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i].flatten()  # flatten to 1D to handle edge cases
            unique_vals, counts = np.unique(votes, return_counts=True)
            final.append(unique_vals[np.argmax(counts)])
        return np.array(final)

    def predict_probs(self, X, classes):
        """
        Averages prob vectors across all trees for soft probability estimates.
        Individual trees return hard 0/1 vectors, but averaging across many trees
        gives meaningful probabilities — e.g. 70/100 trees predict resting CD4+ → 0.70.
        """
        all_probs = np.array([tree.predict_probs(X, classes) for tree in self.trees])
        return np.mean(all_probs, axis=0)