import json as js
import numpy as np
import matplotlib.pyplot as plt

demo = True # set to false to run on full dataset and generate final results for paper (note: will take a few minutes to load on HPC cluster due to large size)
results_dir = "sample_data/sample_results" if demo else "results"

def load_results(model_name):
    with open(f"{results_dir}/{model_name}_results.json", "r") as f:
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
        ax.set_xlabel("FPR", fontsize=20)
        ax.set_ylabel("TPR", fontsize=20)
        ax.set_ticks(fontsize=15)
        ax.legend(fontsize=25)

    plt.suptitle("ROC Curves by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("{results_dir}/roc_curves.png", dpi=300, bbox_inches="tight")
    plt.show()

if not demo:
    y_test = np.load(f"{results_dir}/y_test.npy", allow_pickle=True)
    probs_dict = {"KNN": np.load(f"{results_dir}/knn_probs.npy"),
                  "RF": np.load(f"{results_dir}/rf_probs.npy"),
                  "FFNN": np.load(f"{results_dir}/ffnn_probs.npy")}
    classes = results["KNN"]["classes"]
    plot_roc_grid(y_test, probs_dict, classes)

# F1 Bar Chart
def plot_per_class_f1(results):
    first_model = list(results.keys())[0]
    classes = results[first_model]["classes"]
    model_names = list(results.keys())
    x = np.arange(len(classes))
    width = 0.8 / len(model_names)

    plt.figure(figsize=(14, 7))

    for i, model_name in enumerate(model_names):
        f1_scores = [results[model_name]["per_class"][cell_type]["f1"] for cell_type in classes]
        offset = (i - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, f1_scores, width, label=model_name)

    plt.xticks(x, classes, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Cell Type", fontsize=18)
    plt.ylabel("F1 Score", fontsize=18)
    plt.title("F1 Score by Cell Type Across Classifiers", fontsize=18, fontweight="bold", pad=40)
    plt.tick_params(axis='both', labelsize=15)
    plt.legend(fontsize=15, loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=len(model_names))
    plt.tight_layout()
    plt.savefig(f"{results_dir}/f1_by_cell_type.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_per_class_f1(results)

# Confusion Matrix Heatmaps
def plot_all_confusion_heatmaps(results):
    model_names = list(results.keys())
    short_labels = ["Act. CD4+", "Act. CD8+", "Rest. CD4+", "Rest. CD8+"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for ax, model_name in zip(axes, model_names):
        conf_matrix = np.array(results[model_name]["confusion_matrix"])

        im = ax.imshow(conf_matrix, interpolation="nearest")
        ax.set_title(f"Confusion Matrix - {model_name}")

        tick_marks = np.arange(len(short_labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(short_labels, rotation=30, ha="right")
        ax.set_yticklabels(short_labels)
        ax.set_xlabel("Predicted Cell Type")
        ax.set_ylabel("True Cell Type")

        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center")

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.35)
    plt.savefig(f"{results_dir}/confusion_matrix_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_all_confusion_heatmaps(results)
