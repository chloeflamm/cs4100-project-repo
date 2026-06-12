import loompy
import pandas as pd
import numpy as np

loom_path = "loom_files/t-cell-activation-human-blood-10XV2.loom"
metadata_path = "metadata_files/TCellActivation-Blood-10x_cell_type_2020-03-11.csv"

out_loom = "tcell_blood_sample.loom"
out_metadata = "tcell_blood_metadata_sample.csv"

n_cells = 500 # there may be slightly over this number of cells for duplicate barcodes/samples
random_seed = 42

metadata = pd.read_csv(metadata_path)
metadata["barcode"] = metadata["barcode"].astype(str)

with loompy.connect(str(loom_path), mode="r") as ds:
    loom_cell_names = np.array(ds.ca["cell_names"]).astype(str)

    # Loom cell names look like "CACATTTAGTGGAGAA-0"
    # Metadata barcodes look like "CACATTTAGTGGAGAA"
    loom_barcodes = np.array([cell.split("-")[0] for cell in loom_cell_names])

    # find shared barcodes 
    shared_barcodes = np.intersect1d(loom_barcodes, metadata["barcode"].values)

    # check that number of cells matches number of rows
    print(f"Loom cells: {len(loom_barcodes)}")
    print(f"Metadata rows: {len(metadata)}")
    print(f"Matching barcodes: {len(shared_barcodes)}")

    # randomly sample n_cells from the shared set
    random_gen = np.random.default_rng(random_seed)
    selected_barcodes = random_gen.choice(
        shared_barcodes,
        size=min(n_cells, len(shared_barcodes)),
        replace=False,
    )

    # find indices to preserve original loom ordering
    selected_indices = np.where(np.isin(loom_barcodes, selected_barcodes))[0]

    # slice out selected cells from full expression matrix
    matrix_subset = ds[:, selected_indices]
    row_attrs = {key: ds.ra[key] for key in ds.ra.keys()}
    col_attrs = {key: ds.ca[key][selected_indices] for key in ds.ca.keys()}

    loompy.create(
        filename=str(out_loom),
        layers=matrix_subset,
        row_attrs=row_attrs,
        col_attrs=col_attrs,
    )

# extract barcodes in the same order as lomm file's columns using selected_indices
sampled_cell_names = loom_cell_names[selected_indices]
sampled_barcodes = np.array([cell.split("-")[0] for cell in sampled_cell_names])

# reorder metadata to match loom file's cell order, then save
subset_metadata = metadata.set_index("barcode").loc[sampled_barcodes].reset_index()
subset_metadata.to_csv(out_metadata, index=False)
