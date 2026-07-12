# BA-GNN Supplementary Materials

This repository provides supplementary materials for the manuscript on boundary-aware graph neural network-based airfoil flow-field prediction.

The repository includes the fixed dataset split table, dataset-construction code, data-format description, and access information for the full dataset.

The full implementation and trained weights of the proposed BA-GNN model are not included in this repository.

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
├── full_dataset_link.md
└── README.md
```
## Dataset Split
The file dataset_split/data_split_table.csv provides the fixed dataset split used in the experiments.

All split information is organized in a single table. The column split indicates whether each sample belongs to the training set, validation set, or independent test set.

The split was performed at the airfoil-geometry level. Therefore, all angle-of-attack cases corresponding to the same airfoil are assigned to the same subset, preventing geometric overlap among the training, validation, and independent test sets.

For data-processing convenience, the prefix NACA was added to some internal airfoil filenames, such as NACAGOE. This prefix is only an internal filename convention and does not indicate the actual airfoil family.
## Dataset-Construction Code
The file dataset_creation_code/dataset_generation.py provides the dataset-construction procedure used in this study.

The script includes the main steps for converting CFD mesh and flow-field data into graph-structured samples, including mesh-to-graph conversion, boundary-type identification, node-feature construction, edge-feature construction, and data preprocessing.
## Data Format
Each graph sample contains node features, graph connectivity, edge attributes, and target flow-field variables.

The node features include spatial coordinates, node-type information, and operating-condition information.

The edge attributes include normalized relative displacement, normalized edge
length, and the unit direction vector. The edge type is stored separately in
the `edge_type` tensor.

The target variables are the pressure coefficient Cp and the velocity components U and V.
A detailed description of the graph-data format is provided in dataset_description/data_format_description.md.
## Full Dataset
Due to the large file size, the full graph dataset is provided through the external link listed in full_dataset_link.md.
