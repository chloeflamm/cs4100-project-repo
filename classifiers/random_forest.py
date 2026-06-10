import numpy as np


# a node is either a question (split) or a final answer (leaf)
class Node:
    def __init__(self):
        self.feature   = None   # which column to split on
        self.threshold = None   # the value to compare against
        self.left      = None   # samples <= threshold go here
        self.right     = None   # samples > threshold go here
        self.label     = None   # only set if this is a leaf


# decision tree, RF will make a lot of these when running
class DecisionTree:
    def __init__(self, max_depth=10, n_features=None):
        self.max_depth  = max_depth
        self.n_features = n_features
        self.root       = None

    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)

    def predict(self, X):
        results = []
        for sample in X:
            results.append(self.traverse(sample, self.root))
        return np.array(results)

    def build_tree(self, X, y, depth):
        num_samples = X.shape[0]
        num_classes = len(np.unique(y))

        # stop if we hit max depth or all samples are the same class
        if depth >= self.max_depth or num_classes == 1 or num_samples < 2:
            leaf = Node()
            leaf.label = self.get_majority_class(y)
            return leaf

        # only look at a random subset of features at each split
        all_features    = list(range(X.shape[1]))
        chosen_features = np.random.choice(all_features, size=self.n_features, replace=False)

        best_feature, best_threshold = self.find_best_split(X, y, chosen_features)

        # if we couldnt find a good split just return a leaf
        if best_feature is None:
            leaf = Node()
            leaf.label = self.get_majority_class(y)
            return leaf

        left_idx  = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] >  best_threshold

        node           = Node()
        node.feature   = best_feature
        node.threshold = best_threshold
        node.left      = self.build_tree(X[left_idx],  y[left_idx],  depth + 1)
        node.right     = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        return node

    def find_best_split(self, X, y, features_to_try):
        best_gini      = 999999
        best_feature   = None
        best_threshold = None

        for feature in features_to_try:
            for threshold in np.unique(X[:, feature]):
                left_y  = y[X[:, feature] <= threshold]
                right_y = y[X[:, feature] >  threshold]

                if len(left_y) == 0 or len(right_y) == 0:
                    continue

                gini = self.weighted_gini(left_y, right_y)

                if gini < best_gini:
                    best_gini      = gini
                    best_feature   = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    # gini measures how mixed the classes are for T-Cells, lower the number the better
    def gini(self, y):
        if len(y) == 0:
            return 0
        score = 1.0
        for c in np.unique(y):
            p = np.sum(y == c) / len(y)
            score -= p ** 2
        return score

    def weighted_gini(self, left_y, right_y):
        total = len(left_y) + len(right_y)
        return (len(left_y) / total) * self.gini(left_y) + \
               (len(right_y) / total) * self.gini(right_y)

    def traverse(self, sample, node):
        if node.label is not None:
            return node.label
        if sample[node.feature] <= node.threshold:
            return self.traverse(sample, node.left)
        return self.traverse(sample, node.right)

    def get_majority_class(self, y):
        counts = {}
        for label in y:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
        return max(counts, key=counts.get)


# RF: just a bunch of trees voting together
class RandomForest:
    def __init__(self, n_trees=100, max_depth=10):
        self.n_trees   = n_trees
        self.max_depth = max_depth
        self.trees     = []

    def fit(self, X, y):
        self.trees = []
        # sqrt of total features is the standard amount per split
        n_features_per_split = int(np.sqrt(X.shape[1]))

        for i in range(self.n_trees):
            # bootstrap: sample the data randomly with replacement
            idx      = np.random.choice(len(X), size=len(X), replace=True)
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
            votes               = all_preds[:, i]
            unique_vals, counts = np.unique(votes, return_counts=True)
            final.append(unique_vals[np.argmax(counts)])

        return np.array(final)
