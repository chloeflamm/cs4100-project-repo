"""
Author: Chloe Flamm
This file contains uses a pretrained classifier for cell typing. 
This is intended to  be used as a baseline for our classifiers to compare against a model specialized/trained for cell typing.

CellTypist is a pretrained classifer that uses logistic regression to predict cell types from single cell RNA-seq data.
The package "Immune_All_low.pkl" model, covers a more broad range of cell-types, and more specifically immune subtypes.
If our classifiers can perform comparably to this model, it would be a good indication that our classifiers are learning
meaningful patterns in the data. Otherwise, this could indicate that there is a domain-specific aspect of the data
that our classifiers are not capturing, and may need more biological context and/or specialized features to perform at par. 
"""

# imports
import loompy
import scanpy as sc
import pandas as pd
import numpy as np
import celltypist
from celltypist import models

# Note: CellTypist expects raw data, so we will not use the load_data function from sample_dataloader.py, 
# which performs normalization and log transformation.

def run_celltypist(loom_path, metadata):
    
    # Using same data loading approach as sample_dataloader.py to ensure we get the same matched cells and labels
    # (without normalization or PCA, so I used) 
    metadata = pd.read_csv(metadata)

    with loompy.connect(loom_path) as ds:
        cell_ids = ds.ca["CellID"]

        gene_names = ds.ra["Gene"].astype(str)
        target_barcodes = set(metadata["barcode"].values)
        indices = [i for i, name in enumerate(cell_ids) if name in target_barcodes]
        
        X_raw = ds[:, indices].T
        matched_barcodes = [cell_ids[i] for i in indices]

    # Align labels
    barcode_to_label = dict(zip(metadata["barcode"], metadata["annotated_cell_identity.text"]))
    y = np.array([barcode_to_label[b] for b in matched_barcodes])

    # CellTypist expects an Annadata object as input, so we need to convert
    adata = sc.AnnData(X_raw) # expression matrix (cells x genes)
    adata.var_names = gene_names # gene columns

    # CellTypist requires log-normalized (log1p) expression values
    sc.pp.normalize_total(adata, target_sum=1e4) # normalize so total expression sums to 1e4
    sc.pp.log1p(adata) # log1p 

    # load pretrained CellTypist model
    models.download_models(force_update=False, model="Immune_All_Low.pkl")

    # Run classifier
    # majority voting groups cells into small clusters based on similarity,
    # and then individual prediciton within clusters and assigns predictions.
    # (biologically similar cells should share similar expression pattterns)
    predictions = celltypist.annotate(
        adata,
        model="Immune_All_Low.pkl",
        majority_voting=True 
    )
    
    y_pred = predictions.predicted_labels["majority_voting"].values

    # After running, using the predicted labels, 
    # I was able to map to the labels in our dataset for evaluation
    
    # Note: CellTypist's model has more specific subtypes of T cells, 
    # so I mapped those to the more general "activated" vs "resting" CD4+ and CD8+ T cell labels 
    # in our dataset. This needs a second look - it could be causing a bottleneck if I was wrong
    
    label_map = {
    "Tcm/Naive helper T cells": "resting CD4+ T cell",
    "Tem/Effector helper T cells": "activated CD4+ T cell",
    "Tcm/Naive cytotoxic T cells": "resting CD8+ T cell",
    "Tem/Temra cytotoxic T cells": "activated CD8+ T cell",
    "Tem/Trm cytotoxic T cells": "activated CD8+ T cell",
    "Regulatory T cells": "resting CD4+ T cell",
    "T(agonist)": "activated CD4+ T cell",
    "NK cells": None}

    """
    AI Usage comment: 
    Prompt: how do I create a mask for the labels in my dataset 
    and filter out unmapped cells?
    """
    # Map predictions to your label set, filter out unmapped cells (NK cells)
    mask = np.array([label_map.get(p) is not None for p in y_pred])
    y = y[mask]
    y_pred = np.array([label_map[p] for p in y_pred[mask]])
    y_prob = predictions.probability_matrix.values[mask]

    return y, y_pred, y_prob



