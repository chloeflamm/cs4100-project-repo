import numpy as np
from sample_dataloader import load_data
from knn_classifier import KNN
from evaluate import cross_validate, print_full_report

X, y = load_data("sample_data/tcell_blood_sample.loom",
                 "sample_data/tcell_blood_metadata_sample.csv")

classes = np.unique(y)
print("Classes:", classes)

# Train/test split (80/20)
np.random.seed(42)
idx = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]

# KNN Classifier
print("\nKNN Classifier Results")
print("Cross Validation:")
cross_validate(KNN, X, y, n_folds=5, k=5)

knn = KNN(k=5)
knn.fit(X_train, y_train)
knn_preds = knn.predict(X_test)
print_full_report(y_test, knn_preds, None, classes)
