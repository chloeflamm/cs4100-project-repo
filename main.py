"""
main.py
Authors: Nasir Stanley, Chloe Flamm

Runs three classifiers and saves each result to results/ folder.
- KNN, RF, FFNN: trained on full dataset using Northeastern HPC Explorer Cluster

Results saved to:
    - results/knn_results.json
    - results/rf_results.json
    - results/ffnn_results.json
"""
import os
import json
from evaluate import *
from classifiers.knn_classifier import KNN
from classifiers.ffnn import FFNN
from classifiers.random_forest import RandomForest
from dataloader import load_data
from sample_data.sample_dataloader import load_data_sample

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
'''
# Uncomment for demo with sample data
X, y = load_data_sample(
    "sample_data/tcell_blood_sample.loom",
    "sample_data/tcell_blood_metadata_sample.csv"
)
'''
def save_results(name, results_dict):
    '''Save a single classifier's results to results/{name}_results.json.'''
    path = os.path.join("results", f"{name}_results.json")
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"Saved to {path}")

# Load full dataset for remaining classifiers
# Note: this will take a few minutes to load on HPC cluster due to large size
# Comment this out in order to run the demonstration with the sample data above
X, y = load_data(
    loom_dir=os.path.join(BASE_DIR, "loom_files/"),
    csv_dir=os.path.join(BASE_DIR, "metadata_files/"))

classes = np.unique(y)
print("Classes:", classes)

# Train/test split — 80/20, fixed seed for reproducibility
np.random.seed(42)
idx   = np.random.permutation(len(X))
split = int(0.8 * len(X))
X_train, X_test = X[idx[:split]], X[idx[split:]]
y_train, y_test = y[idx[:split]], y[idx[split:]]


# KNN Classifier
print("\nKNN Classifier Results")
print("-"*30)
print("Cross Validation:")
cross_validate(KNN, X, y, n_folds=5, k=5)

knn = KNN(k=5)
knn.fit(X_train, y_train)
knn_preds, knn_probs = knn.predict(X_test)
print_full_report(y_test, knn_preds, knn_probs, classes)

knn_aucs = auc_roc(y_test, knn_probs, classes)
save_results("knn", {
    "classes": classes.tolist(),
    "accuracy": float(accuracy(y_test, knn_preds)),
    "macro_f1": float(macro_f1(y_test, knn_preds, classes)),
    "per_class": {c: {k: float(v) for k, v in s.items()} for c, s in precision_recall_f1(y_test, knn_preds, classes).items()},
    "confusion_matrix": confusion_matrix(y_test, knn_preds, classes).tolist(),
    "auc_roc": {c: float(v) for c, v in knn_aucs.items()}
})

# Random Forest Classifier
print("\nRandom Forest Classifier Results:")
print("-"*30)
print("Cross Validation:")
cross_validate(RandomForest, X, y, n_folds=5, n_trees=100, max_depth=10)

rf = RandomForest(n_trees=100, max_depth=10)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_probs = rf.predict_probs(X_test, classes)
print_full_report(y_test, rf_preds, rf_probs, classes)

save_results("rf", {
    "classes": classes.tolist(),
    "accuracy": float(accuracy(y_test, rf_preds)),
    "macro_f1": float(macro_f1(y_test, rf_preds, classes)),
    "per_class": {c: {k: float(v) for k, v in s.items()} for c, s in precision_recall_f1(y_test, rf_preds, classes).items()},
    "auc_roc": auc_roc(y_test, rf_probs, classes),
    "confusion_matrix": confusion_matrix(y_test, rf_preds, classes).tolist()
})

# FFNN Classifier
print("FFNN Classifier Results")
print("-"*30)

ffnn = FFNN(input_dim=X_train.shape[1], hidden_dim=256, output_dim=len(classes), lr=0.01, epochs=100)
ffnn.fit(X_train, y_train)
ffnn_preds, ffnn_probs = ffnn.predict(X_test)
print_full_report(y_test, ffnn_preds, ffnn_probs, classes)

ffnn_aucs = auc_roc(y_test, ffnn_probs, classes)
save_results("ffnn", {
    "classes": classes.tolist(),
    "accuracy": float(accuracy(y_test, ffnn_preds)),
    "macro_f1": float(macro_f1(y_test, ffnn_preds, classes)),
    "per_class": {c: {k: float(v) for k, v in s.items()} for c, s in precision_recall_f1(y_test, ffnn_preds, classes).items()},
    "confusion_matrix": confusion_matrix(y_test, ffnn_preds, classes).tolist(),
    "loss_history": [float(x) for x in ffnn.loss_history],
    "auc_roc": {c: float(v) for c, v in ffnn_aucs.items()}
})


# Raw ROC Classifier Probabilities for plotting
np.save("results/y_test.npy", y_test)
np.save("results/knn_probs.npy", knn_probs)
np.save("results/rf_probs.npy", rf_probs)
np.save("results/ffnn_probs.npy", ffnn_probs)