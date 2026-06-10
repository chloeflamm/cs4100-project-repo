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
        # added max thresholds to limit number of threshold candidates considered per feature for speed
        # otherwise would check every unique value in each feature, which is slow
        self.max_thresholds = max_thresholds 
        self.root = None

    def fit(self, X, y):
        """builds tree recursively starting from the root node"""
        self.root = self.build_tree(X, y, depth=0)

    def predict(self, X):
        """route each sample through the tree to get a predicted label"""
        results = []
        for sample in X:
            results.append(self.traverse(sample, self.root))
        return np.array(results)

    def build_tree(self, X, y, depth):
        """recursively builds the tree by finding the best split at each node until stopping conditions are met"""
        num_samples = X.shape[0]
        num_classes = len(np.unique(y))

        # stop if we hit max depth or all samples are the same class (no split needed)
        if depth >= self.max_depth or num_classes == 1 or num_samples < 2: # fewer than two can't split
            leaf = Node()
            leaf.label = self.get_majority_class(y)
            return leaf

        # only look at a random subset of features at each split
        # key to RF randomness, decorrelating trees and improving emsemble performance
        all_features = list(range(X.shape[1]))
        chosen_features = np.random.choice(all_features, size=self.n_features, replace=False)

        # find the best feature/threshold combination among chosen features
        best_feature, best_threshold = self.find_best_split(X, y, chosen_features)

        # if we couldnt find a good split just return a leaf (if all fts are the same)
        if best_feature is None:
            leaf = Node()
            leaf.label = self.get_majority_class(y)
            return leaf

        # split into left and right branches
        left_idx  = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        # create an internal node and recursively build left/right subtrees
        node = Node()
        node.feature = best_feature
        node.threshold = best_threshold
        node.left = self.build_tree(X[left_idx],  y[left_idx],  depth + 1)
        node.right = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return node

    def find_best_split(self, X, y, features_to_try):
        """finds the best feature/threshold to split on among the given features by minimizing gini impurity"""
        best_gini = 999999
        best_feature = None
        best_threshold = None

        for feature in features_to_try:
            thresholds = np.unique(X[:, feature])
            # limit the number of thresholds considered for speed
            # randomly sample if there are too many threshold candidates
            if len(thresholds) > self.max_thresholds:
                thresholds = np.random.choice(thresholds, size=self.max_thresholds, replace=False)
            for threshold in thresholds:
                left_y  = y[X[:, feature] <= threshold]
                right_y = y[X[:, feature] > threshold]

                # skip split that don't divide data
                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = self.weighted_gini(left_y, right_y)

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # gini measures how mixed the classes are for T-Cells, lower the number the better
    def gini(self, y):
        """measures class impurity, 0 means all are same class (pure),
        higher values mean more mixed. Lower is better, since we want splits
        that separate cell types well."""
        if len(y) == 0:
            return 0
        score = 1.0
        for c in np.unique(y):
            p = np.sum(y == c) / len(y)
            score -= p ** 2
        return score

    def weighted_gini(self, left_y, right_y):
        """weights each child's gini by its proportion of total samples,
        ensures larger splits are penalized more if they are impure."""
        total = len(left_y) + len(right_y)
        return (len(left_y) / total) * self.gini(left_y) + \
               (len(right_y) / total) * self.gini(right_y)

    def traverse(self, sample, node):
        """return label if we reach a leaf, otherwise keep traversing"""
        if node.label is not None:
            return node.label
        if sample[node.feature] <= node.threshold: # left is less than thres
            return self.traverse(sample, node.left)
        return self.traverse(sample, node.right) # right if greater than thres

    def get_majority_class(self, y):
        """returns the most freq class label in y"""
        values, counts = np.unique(y, return_counts=True) # edit: replaced using numpy functionality
        return values[np.argmax(counts)]
    
    def predict_probs(self, X, classes):
        """
        Traverses the tree for each sample and returns a one-hot probability 
        vector instead of a single label (useing traverse_probs). 
        Returns an (n_samples x n_classes) matrix.
        """
        results = []
        for sample in X:
            results.append(self.traverse_probs(sample, self.root, classes))
        return np.array(results)

    def traverse_probs(self, sample, node, classes):
        """
        Follows the same logic as traverse() but returns a prob
        vector at each leaf: 1 leaf's class, 0 for others.
        Single trees are deterministic so probs are always 0 or 1.
        """
        if node.label is not None:
            # initialize probs to 0 for all classes
            prob = np.zeros(len(classes))
            # find the index of the leaf's class in array
            idx = np.where(classes == node.label)[0]
            if len(idx) > 0:
                prob[idx[0]] = 1
            return prob
        # if not a leaf, split and continue traversing (same as traverse())
        # left if <= threshold, right > threshold
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
        # sqrt of total features is the standard amount per split
        n_features_per_split = int(np.sqrt(X.shape[1]))

        for i in range(self.n_trees):
            # bootstrap: sample the data randomly with replacement
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sample = X[idx]
            y_sample = y[idx]

            tree = DecisionTree(max_depth=self.max_depth, n_features=n_features_per_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # get predictions from every tree
        all_preds = np.array([tree.predict(X) for tree in self.trees])

        # majority vote across all trees for each sample
        final = []
        for i in range(X.shape[0]):
            votes = all_preds[:, i]
            unique_vals, counts = np.unique(votes, return_counts=True)
            final.append(unique_vals[np.argmax(counts)])

        return np.array(final)
    
    def predict_probs(self, X, classes):
        """
        Avgs prob vectors across all trees to produce soft probs.
        Individual trees return hard one-hot vectors, but avging across many trees
        converts these into meaningful prob estimates.
        for ex: if 70/100 trees predict "resting CD4+", that class gets 0.70.
        """
        all_probs = np.array([tree.predict_probs(X, classes) for tree in self.trees])
        return np.mean(all_probs, axis=0)
