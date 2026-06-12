import json
import numpy as np
import matplotlib.pyplot as plt

def load_results(model_name):
    with open(f"results/{model_name}_results.json", "r") as f:
        return json.load(f)

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


# Grouped Bar Chart (F1 score by Cell Type/Classifier)
def plot_per_class_f1(results):
    # class order from first model
    first_model = list(results.keys())[0]
    classes = results[first_model]["classes"]

    # names of each classifier
    model_names = list(results.keys())

    # one pos. per cell type
    x = np.arange(len(classes))
    width = 0.8 / len(model_names) # width so all classifiers fit

    plt.figure(figsize=(14, 7))

    # one bar for each classifier per cell type
    for i, model_name in enumerate(model_names):
        # saved f1 score 
        f1_scores = [
            results[model_name]["per_class"][cell_type]["f1"]
            for cell_type in classes
        ]
        # make bars side-by-side
        offset = (i - (len(model_names) - 1) / 2) * width
        plt.bar(x + offset, f1_scores, width, label=model_name)

    plt.xticks(x, classes, rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.xlabel("Cell Type")
    plt.ylabel("F1 Score")
    plt.title("F1 Score by Cell Type Across Classifiers")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/f1_by_cell_type.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_per_class_f1(results)


# Confusion Matrix Heatmaps
def plot_all_confusion_heatmaps(results):
   # names of each classifier
    model_names = list(results.keys())

    # shortened labels for readability
    short_labels = [
        "Act. CD4+",
        "Act. CD8+",
        "Rest. CD4+",
        "Rest. CD8+"
    ]

    # create 2 x 2 grid of plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    # flatten axes 
    axes = axes.flatten()

    for ax, model_name in zip(axes, model_names):
        # get confusion matrix and class labels 
        conf_matrix = np.array(results[model_name]["confusion_matrix"])
        classes = results[model_name]["classes"]

        im = ax.imshow(conf_matrix, interpolation="nearest")
        ax.set_title(f"Confusion Matrix - {model_name}")

        # labels
        tick_marks = np.arange(len(short_labels))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(short_labels, rotation=30, ha="right")
        ax.set_yticklabels(short_labels)

        # Add axis labels.
        ax.set_xlabel("Predicted Cell Type")
        ax.set_ylabel("True Cell Type")

        # count values inside each cell
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax.text(j, i, conf_matrix[i, j], ha="center", va="center")

        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.35) # need more bottom space for pred. cells
    plt.savefig("results/confusion_matrix_heatmaps.png", dpi=300, bbox_inches="tight")
    plt.show()

plot_all_confusion_heatmaps(results)