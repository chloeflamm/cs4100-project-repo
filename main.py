import numpy as np
import json
from sample_dataloader import load_data
from classifiers.knn_classifier import KNN
from evaluate import cross_validate, print_full_report, accuracy, macro_f1, precision_recall_f1, convert_to_serializable

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

# save evaluation metrics results to json
knn_results = {
    "accuracy": float(accuracy(y_test, knn_preds)),
    "macro_f1": float(macro_f1(y_test, knn_preds, classes)),
    "per_class": precision_recall_f1(y_test, knn_preds, classes)
}
with open("sample_data/sample_results/knn_results.json", "w") as f:
    json.dump(knn_results, f, default=convert_to_serializable)

#Hyperparam tune KNN classifier
# Pretuning results:
#KNN Classifier Results:
#Accuracy: 0.7722772277227723
#F1 Scores: {np.str_('activated CD4+ T cell'): np.float64(0.7741935431581685), np.str_('activated CD8+ T cell'): np.float64(0.6999999945000001), np.str_('resting CD4+ T cell'): np.float64(0.8181818131224173), np.str_('resting CD8+ T cell'): np.float64(0.6874999947460939)}
#Confusion Matrix:
#     AC4 AC8 RC4 RC8
# AC4[[24  1  7  1]
# AC8[ 5  7  0  0]
# RC4[ 0  0 36  1]
# RC8[ 0  0  8 11]]
# Activated CD4+ — 24 correct, 7 confused with resting CD4+
# Activated CD8+ — 7 correct, 5 confused with activated CD4+ (small class, not enough examples)
# Resting CD4+ — 36 correct, nearly perfect
# Resting CD8+ — 11 correct, 8 confused with resting CD4+

# Problem: sturggling defining active vs resting due to similar gene expression