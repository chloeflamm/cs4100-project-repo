import json as js
import numpy as np
import matplotlib.pyplot as plt

def load_results(model_name):
    with open(f"results/{model_name}_results.json", "r") as f:
        return js.load(f)

results = {"KNN": load_results("knn"),
           "RF": load_results("rf"),
           "FFNN": load_results("ffnn"),
           "CellTypist": load_results("celltypist")}

def plot_roc_grid(y_test, probs_dict, classes):
    """
    probs_dict: {"KNN": np.array, "RF": np.array, "FFNN": np.array}
    one subplot per class with one curve per model
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    colors = {"KNN": "#4C72B0", "RF": "#55A868", "FFNN": "#C44E52"}

    for i, c in enumerate(classes):
        ax = axes[i]
        # converts multiclass labels to binary for each class on iteration
        tru_binary = (np.array(y_test) == c).astype(int)

        for model_name, probs in probs_dict.items():
            scores = probs[:, i] # extract predicted probabilities for class c
            sorted_idx = np.argsort(scores)[::-1]
            binary_sorted = tru_binary[sorted_idx]
            tp = np.cumsum(binary_sorted)
            fp = np.cumsum(1 - binary_sorted)
            tpr = tp / (tp[-1] + 1e-8)
            fpr = fp / (fp[-1] + 1e-8)
            auc = abs(np.trapezoid(tpr, fpr))
            ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.3f})", color=colors[model_name])

        # Plot configurations
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_title(c, fontsize=11)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(fontsize=8)

    plt.suptitle("ROC Curves by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/roc_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

# Load saved raw probability graphs
y_test = np.load("results/y_test.npy", allow_pickle=True)
probs_dict = {"KNN": np.load("results/knn_probs.npy"),
              "RF": np.load("results/rf_probs.npy"),
              "FFNN": np.load("results/ffnn_probs.npy")}

classes = results["KNN"]["classes"]
plot_roc_grid(y_test, probs_dict, classes)
