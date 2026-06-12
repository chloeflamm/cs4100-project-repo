# A Comparative Analysis of T-Cell State Classification from Single-cell RNA-seq Data

#### Team Members: Chloe Flamm, Nasir Stanley, Jasdeep Singh

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

```text
pip install -r requirements.txt
```

[Note] This dataset is extensive in size, and required jobs submitted to Northeastern's High Performance Computer (HPC) cluster for training. 

### III. Instructions to execute

-  ```text
    pip install -r requirements.txt
    ```
- Dataset Download: 
    - Human Cell Atlas Data Explorer: https://explore.data.humancellatlas.org/projects/4a95101c-9ffc-4f30-a809-f04518a23803/get-curl-command
    - Species: Homo sapiens
    - File Type: csv, loom
    - Shell: Bash
    - (Request curl Command)
- Place LOOM files in loom_file/ and metadata in metadata_files/
- For sample data, use either sample_data/tcell_blood_metadata_sample.csv and sample_data/tcell_blood_sample.loom OR run:
  ```text
    python sampledata.py
    ```
- ```text
  python main.py
  ```
  This runs from classifiers from scratch and saves results to JSON files in results/ folder.
- To run pretrained CellTypist baseline on the raw expression data and save to results/,
  run:
  ```text
  python pretrained_celltypist.py
  ```
- To generate visualizations (F1 score by cell type across classifiers, Confusion matrix heatmaps, ROC curves) after JSON results have been saved,
  run:
  ```text
  python visualize.py.
  ```
- [Note] The four target labels used for classification are:
    - activated CD4+ T cell
    - activated CD8+ T cell
    - resting CD4+ T cell
    - resting CD8+ T cell
- Unassigned labels are excluded from the main classification task so that all classifiers are evaluated on the same four-class problem.

### IV. Organization of Code 


```text
cs4100-project-repo/
├── classifiers/                 # Implemented classifiers
│   ├── ffnn.py                  # Feed-forward neural network classifier using PyTorch
│   ├── knn.py                   # K-nearest neighbors classifier
│   └── rf.py                    # Random forest classifier 
│
├── loom_files/                  # LOOM gene-expression data files
├── metadata_files/              # Metadata CSV files
│
├── results/                     # Saved model outputs and generated figures
│   ├── ffnn_results.json        # Evaluation metrics for the feed-forward neural network
│   ├── knn_results.json         # Evaluation metrics for the KNN classifier
│   ├── rf_results.json          # Evaluation metrics for the random forest classifier
│   ├── celltypist_results.json  # Evaluation metrics for the CellTypist baseline
│   └── ...                      # Saved visualization PNGs
│
├── dataloader.py                # Loads, matches, preprocesses, and splits expression/metadata files
├── evaluate.py                  # Shared evaluation metrics: accuracy, F1, confusion matrix, AUC-ROC
├── main.py                      # Main training/evaluation pipeline for project classifiers
├── pretrained_celltypist.py     # Runs the pretrained CellTypist baseline and maps labels
└── visualization.py             # Generates result visualizations from saved JSON files
