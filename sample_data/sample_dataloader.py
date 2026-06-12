import loompy
import pandas as pd
import numpy as np

def pca(X, n_components=50):
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort by largest eigenvalue
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    components = eigenvectors[:, :n_components]
    X_pca = X_centered @ components
    return X_pca

def load_data(loom_path, csv_path, n_top_genes=500, n_pca_components=50):
    metadata = pd.read_csv(csv_path)

    with loompy.connect(loom_path) as ds:
        cell_ids = ds.ca["CellID"]

        target_barcodes = set(metadata["barcode"].values)
        indices = [i for i, name in enumerate(cell_ids) if name in target_barcodes]
        print(f"Matched {len(indices)} cells")

        X = ds[:, indices].T
        matched_barcodes = [cell_ids[i] for i in indices]

    # Align labels
    barcode_to_label = dict(zip(metadata["barcode"], metadata["annotated_cell_identity.text"]))
    y = np.array([barcode_to_label[b] for b in matched_barcodes])

    # Keep top genes by variance
    variances = np.var(X, axis=0)
    top_genes = np.argsort(variances)[::-1][:n_top_genes]
    X = X[:, top_genes]

    # Normalize
    X = np.log1p(X)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # PCA
    print(f"Running PCA: {X.shape[1]} genes -> {n_pca_components} components")
    X = pca(X, n_components=n_pca_components)
    print(f"Final shape: {X.shape}")

    return X, y
