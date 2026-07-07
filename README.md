# BA-GNN Supplementary Materials

This repository provides supplementary materials for the manuscript on boundary-aware graph neural network-based airfoil flow-field prediction.

The repository mainly includes the dataset split table, dataset-construction code, data-format description, and supplementary numerical result summaries.

The full implementation of the proposed BA-GNN model is not included in this repository.

## Repository Structure

```text
BA-GNN/
├── dataset_split/
│   └── data_split_table.csv
│
├── dataset_creation_code/
│   └── dataset_generation.py
│
├── dataset_description/
│   └── data_format_description.md
│
├── results_summary/
│   └── experimental_results_summary.csv
│
├── full_dataset_link.md
└── README.md
```
## Dataset Split
The file dataset_split/data_split_table.csv provides the fixed dataset split used in the experiments.
Instead of using separate files for the training, validation, test, and independent-test sets, all split information is organized in a single table. The column split indicates whether each sample belongs to the training set, validation set, test set, or independent test set.
## Dataset-Construction Code
The file dataset_creation_code/dataset_generation.py provides the dataset-construction procedure used in this study.
The script includes the main steps for converting CFD mesh and flow-field data into graph-structured data, including mesh-to-graph conversion, boundary-type identification, node and edge feature construction, and data preprocessing.
## Data Format
Each graph sample contains node features, edge attributes, and target flow-field variables.
The node features include spatial coordinates, node type, and operating-condition information.
The edge attributes include relative displacement, edge length, direction vector, and edge type.
The target variables include pressure coefficient and velocity components.
## Full Dataset
Due to the large file size, the full dataset is provided through an external link in full_dataset_link.md.
