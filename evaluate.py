import numpy as np

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def confusion_matrix(y_true, y_pred, classes):
    n = len(classes)
    cm = np.zeros((n, n), dtype=int)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true]][class_to_idx[pred]] += 1
    return cm

def f1_score(y_true, y_pred, classes):
    scores = {}
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        scores[c] = 2 * precision * recall / (precision + recall + 1e-8)
    return scores

def precision_recall_f1(y_true, y_pred, classes):
    results = {}
    for c in classes:
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        results[c] = {"precision": precision, "recall": recall, "f1": f1}
    return results


def macro_f1(y_true, y_pred, classes):
    scores = precision_recall_f1(y_true, y_pred, classes)
    return np.mean([scores[c]["f1"] for c in classes])

def cross_validate(model_class, X, y, n_folds=5, **model_kwargs):
    n = len(X)
    fold_size = n // n_folds
    indices = np.random.permutation(n)
    accuracies = []
    macro_f1s = []
    classes = np.unique(y)

    for fold in range(n_folds):
        val_idx = indices[fold * fold_size: (fold + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold * fold_size],
                                    indices[(fold + 1) * fold_size:]])

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_kwargs)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        acc = accuracy(y_val, preds)
        mf1 = macro_f1(y_val, preds, classes)
        accuracies.append(acc)
        macro_f1s.append(mf1)
        print(f"  Fold {fold + 1}: Accuracy={acc:.4f}, Macro F1={mf1:.4f}")

    print(f"  Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"  Mean Macro F1: {np.mean(macro_f1s):.4f} ± {np.std(macro_f1s):.4f}")
    return accuracies, macro_f1s

def auc_roc(y_true, y_prob, classes):
    """
    One-vs-rest AUC-ROC for each class.
    y_prob: (n_samples x n_classes) probability matrix
    """
    aucs = {}
    for i, c in enumerate(classes):
        binary_true = (y_true == c).astype(int)
        scores = y_prob[:, i]

        # Sort by score descending
        sorted_idx = np.argsort(scores)[::-1]
        binary_true_sorted = binary_true[sorted_idx]

        # Compute TPR and FPR at each threshold
        tp = np.cumsum(binary_true_sorted)
        fp = np.cumsum(1 - binary_true_sorted)
        tpr = tp / (tp[-1] + 1e-8)
        fpr = fp / (fp[-1] + 1e-8)

        # AUC via trapezoidal rule
        auc = np.trapz(tpr, fpr)
        aucs[c] = abs(auc)  # abs in case of reverse ordering
    return aucs

def print_full_report(y_true, y_pred, y_prob, classes):
    print(f"\nAccuracy: {accuracy(y_true, y_pred):.4f}")
    print(f"Macro F1: {macro_f1(y_true, y_pred, classes):.4f}")

    print("\nPer-class metrics:")
    scores = precision_recall_f1(y_true, y_pred, classes)
    for c in classes:
        s = scores[c]
        print(f"  {c}: Precision={s['precision']:.4f}, Recall={s['recall']:.4f}, F1={s['f1']:.4f}")

    if y_prob is not None:
        aucs = auc_roc(y_true, y_prob, classes)
        print("\nAUC-ROC (one-vs-rest):")
        for c in classes:
            print(f"  {c}: {aucs[c]:.4f}")
        print(f"  Mean AUC: {np.mean(list(aucs.values())):.4f}")

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, classes)
    col_width = max(len(c) for c in classes) + 2
    header = "".join([f"{c:>{col_width}}" for c in classes])
    print(f"{'':>25}{header}")
    for i, c in enumerate(classes):
        row = "".join([f"{cm[i][j]:>{col_width}}" for j in range(len(classes))])
        print(f"{c:>25}{row}")