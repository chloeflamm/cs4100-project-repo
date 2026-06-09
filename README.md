# A Comparative Analysis of T-Cell State Classification from Single-cell RNA-seq Data

#### Team Members: Chloe Flamm, Nasir Stanley

### I. Overview of the Problem

T-cells are a foundational element of the human immune system, and the functional state a T-cell expresses (resting CD4+, activated CD4+, resting CD8+, and activated CD8+) has direct implications for immune response, disease progression, and therapeutic outcomes. Traditional methods for distinguishing T-cell states are labor-intensive and often imprecise, particularly in complex tissue environments such as tumors. 

From the Human Cell Atlas dataset, we downloaded single-cell DNA sequencing data from 50k human T cells across multiple tissues. The dataset includes metadata CSV file contining annotated cell identities. Each cell has a unique barcode that links the metadata label to the corresponding cell in a LOOM expression matrix. The LOOM files contain the cell-by-gene gene-expression values that will be used as input features for our classifiers. While, the annotated identities from the CSV files will serve as the target labels.

This project performs a comparative analysis of multiple classification techniques applied to the single-cell DNA sequencing (scRNA-seq) dataset to predict the activation state of individual T-cells. We implement three classifiers from scratch and evaluate them systematically:

- k-Nearest Neighbors (kNN)
- Random Forest
- Neural Network 

Each classifier takes per-cell gene expression profiles as input features and outputs a predcited T-cell state label. We evaluate performance using cross-validation, confusion matrices, F1-score, and AUC-ROC curves. We then visualized the evaluation metrics, making comparisons between our three classification techniques.

In addition to our from-scratch classifiers, we started by running our data through CellTypist, a specialized pre-trained model for cell type classification to create a baseline for our evaluations. 

### II. Setup Requirements

- Libraries: requirements.txt

- Dataset Download: 
    - Human Cell Atlas Data Explorer: https://explore.data.humancellatlas.org/projects/4a95101c-9ffc-4f30-a809-f04518a23803/get-curl-command
    - Species: Homo sapiens
    - File Type: csv, loom
    - Shell: Bash
    - (Request curl Command)


- Pre-Cleaning
    - Once dataset was downloaded, we selected the LOOM files with names in plain-english that corresponded to the CSV metadata.
    - In LOOM files, cell barcodes are named "cell_name" and have "-0" at the tail end of each entry, which require cleaning. In metadata CSVs, the corresponding barcode column is "barcode".
    - Further downsampling for test data was created by cross-checking barcodes and selecting 500 cells.

- [Note] This dataset is extensive in size, and required jobs submitted to Northeastern's High Performance Computer (HPC) cluster for training. 

### III. Instructions to execute

- Instruction to run main evaluation file


### IV. Organization of Code (In progress)

Data: 
    - loomfiles/
    - metadata/

- Classifiers:
- Evaluation
- Results.txt
