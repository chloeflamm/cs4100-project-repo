"""
dataloader.py
Authors: Nasir Stanley, Chloe Flamm

Loads and preprocesses T-cell scRNA-seq data from multiple LOOM and CSV files
across four tissue types (blood, lung, lymph node, bone marrow).

Pipeline:
    1. Match cell barcodes between LOOM expression matrix and CSV metadata
    2. Filter out unassigned cell labels
    3. Select top 500 highly variable genes
    4. Log-normalize and z-score expression values
    5. Reduce to 50 PCA components for classifier input
"""

import loompy
import pandas as pd
import numpy as np
import os

# Paired LOOM files and their corresponding metadata CSVs
LOOM_CSV_PAIRS = [
    ("t-cell-activation-human-blood-10XV2.loom", "TCellActivation-Blood-10x_cell_type_2020-03-11.csv"),
    ("t-cell-activation-human-lung-10XV2.loom", "TCellActivation-lung-10x_cell_type_2020-03-11.csv"),
    ("t-cell-activation-human-lymph-10XV2.loom", "TCellActivation-lymph-node-10x_cell_type_2020-03-11.csv"),
    ("t-cell-activation-human-hematopoietic-10XV2.loom", "TCellActivation-bone-marrow-10x_cell_type_2020-03-11.csv"),
]

# Cells with no confident annotation — excluded from classification
EXCLUDED_CLASSES = ['unassigned activated T call', 'unassigned resting T cell']


def pca(X, n_components=50):
    """
    PCA implemented from scratch using eigen-decomposition.
    Centers the data, computes covariance matrix, and projects
    onto the top n_components eigenvectors (directions of max variance).
    """
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvectors by descending eigenvalue (largest variance first)
    idx = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, idx][:, :n_components]

    return X_centered @ components

def load_single(loom_path, csv_path):
    """
    Load one LOOM + CSV pair and align cells by barcode.
    Uses vectorized np.isin matching instead of a Python loop
    for significantly faster barcode lookup on large files.
    """

    metadata = pd.read_csv(csv_path)
    barcode_to_label = dict(zip(
        metadata["barcode"],
        metadata["annotated_cell_identity.text"]
    ))
    target_barcodes = set(metadata["barcode"].values)

    with loompy.connect(loom_path) as ds:
        cell_ids = np.array(ds.ca["CellID"])

        # Vectorized barcode matching — much faster than iterating cell by cell
        mask    = np.isin(cell_ids, list(target_barcodes))
        indices = np.where(mask)[0]

        if len(indices) == 0:
            print(f"WARNING: No barcode matches in {os.path.basename(loom_path)}")
            return None, None

        print(f"{os.path.basename(loom_path)}: matched {len(indices)} cells")
        X = ds[:, indices].T  # shape: (cells x genes)
        matched_barcodes = cell_ids[indices]

    y = np.array([barcode_to_label[b] for b in matched_barcodes])
    return X, y


def load_data(loom_dir, csv_dir, n_top_genes=500, n_pca_components=50):
    """
    Load and preprocess all four tissue datasets into a single feature matrix to run on the classifiers.
    Args:
        loom_dir: path to folder containing LOOM files
        csv_dir: path to folder containing metadata CSVs
        n_top_genes: number of most variable genes to keep (default 500)
        n_pca_components: number of PCA components for final feature space (default 50)
    Returns:
        X: (n_cells x n_pca_components) float array
        y: (n_cells,) string array of cell type labels
    """
    all_X, all_y = [], []

    print("Loading files...")
    for loom_fname, csv_fname in LOOM_CSV_PAIRS:
        loom_path = os.path.join(loom_dir, loom_fname)
        csv_path  = os.path.join(csv_dir,  csv_fname)

        if not os.path.exists(loom_path):
            print(f"MISSING LOOM: {loom_fname} — skipping")
            continue
        if not os.path.exists(csv_path):
            print(f"MISSING CSV: {csv_fname} — skipping")
            continue

        X, y = load_single(loom_path, csv_path)
        if X is not None:
            all_X.append(X)
            all_y.append(y)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"\nTotal cells across all tissues: {X.shape[0]}")

    # Remove cells with no bad annotation
    mask = ~np.isin(y, EXCLUDED_CLASSES)
    X, y = X[mask], y[mask]
    print(f"After removing unassigned cells: {X.shape[0]}")

    # Print class distribution
    classes, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for c, n in zip(classes, counts):
        print(f"  {c}: {n} cells")

    # Select top genes by variance to reduce noise and computation
    print(f"\nSelecting top {n_top_genes} genes by variance...")
    top_genes = np.argsort(np.var(X, axis=0))[::-1][:n_top_genes]
    X = X[:, top_genes]

    # Log-normalize then z-score per gene
    print("Normalizing...")
    X = np.log1p(X)
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    # PCA: compress 500 genes -> 50 components
    print(f"Running PCA: {X.shape[1]} genes -> {n_pca_components} components...")
    X  = pca(X, n_components=n_pca_components)
    print(f"Final shape: {X.shape}")

    return X, y