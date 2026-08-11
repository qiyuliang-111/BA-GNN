# BA-GNN Supplementary and Reproducibility Materials

This repository provides supplementary and reproducibility materials for the manuscript on boundary-aware graph neural network (BA-GNN)-based airfoil flow-field prediction.

The released materials are intended to support transparent inspection of the dataset construction, exact data split, model architecture, preprocessing procedure, trained-model inference, and independent-test evaluation reported in the manuscript.

## Repository Contents

This repository provides:

- the exact geometry-level train/validation/test split used in the manuscript;
- the dataset-construction code;
- a detailed description of the graph-data format;
- access information for the full graph dataset;
- the BA-GNN model definition required to load the released checkpoint;
- a trained BA-GNN checkpoint;
- normalization parameters derived from the training set;
- runnable single-case inference code;
- runnable independent-test evaluation code;
- model and experimental configuration information;
- software-environment and dependency information.

The complete training implementation is not publicly released because it contains project-specific modules associated with ongoing research.

## Repository Structure

```text
BA-GNN/
├── checkpoint/
│   └── bagnn_seed0_best.pt
│
├── normalization/
│   ├── normalization_stats.json
│   └── scalers.pkl
│
├── reproducibility/
│   ├── model.py
│   ├── inference.py
│   ├── evaluate.py
│   ├── fixed_split_loader.py
│   ├── fixed_split.json
│   └── test_airfoils.txt
│
├── dataset_split/
│   └── data_split_table.csv
│
├── dataset_creation_code/
│   └── dataset_generation.py
│
├── dataset_description/
│   └── data_format_description.md
│
├── config.yaml
├── requirements.txt
├── environment_info.json
├── full_dataset_link.md
└── README.md
```

## Dataset Split

The experiments use a geometry-level data split to prevent information leakage between different angle-of-attack cases of the same airfoil.

The complete dataset contains 408 airfoil geometries:

- 338 airfoils for training;
- 62 airfoils for validation;
- 8 unseen airfoils reserved for independent testing.

All angle-of-attack cases corresponding to the same airfoil are assigned to the same subset.

The exact split used in the manuscript is provided in:

```text
reproducibility/fixed_split.json
```

The eight independent test airfoils are additionally listed in:

```text
reproducibility/test_airfoils.txt
```

The file

```text
dataset_split/data_split_table.csv
```

provides the corresponding case-level split information.

For data-processing convenience, the prefix `NACA` was added to some internal airfoil filenames. This prefix is only an internal filename convention and does not necessarily indicate the actual airfoil family.

## Dataset Construction

The script

```text
dataset_creation_code/dataset_generation.py
```

contains the graph-dataset construction procedure used in this study.

The main processing steps include:

- CFD mesh-to-graph conversion;
- automatic boundary-type identification;
- node-feature construction;
- edge-feature construction;
- graph-connectivity construction;
- preprocessing and serialization into PyTorch Geometric graph samples.

## Graph Data Format

Each graph sample is stored as a PyTorch Geometric data object and contains the information required by BA-GNN, including:

- node features;
- graph connectivity (`edge_index`);
- continuous edge attributes (`edge_attr`);
- node-type labels (`node_type`);
- edge-type labels (`edge_type`);
- target flow-field variables.

The prediction targets are:

- pressure coefficient, `Cp`;
- streamwise velocity component, `U`;
- transverse velocity component, `V`.

A detailed description of the graph-data format is provided in:

```text
dataset_description/data_format_description.md
```

## Full Dataset

Because of the large size of the complete graph dataset, it is distributed through the external access link provided in:

```text
full_dataset_link.md
```

The released dataset can be used together with the fixed split and reproducibility scripts provided in this repository.

## Released BA-GNN Checkpoint

A trained Full BA-GNN checkpoint is provided at:

```text
checkpoint/bagnn_seed0_best.pt
```

The released checkpoint corresponds to one of the trained models used in the study and is provided for reproducible inference and independent evaluation.

The public model architecture required to load the checkpoint is implemented in:

```text
reproducibility/model.py
```

This file contains the BA-GNN architecture required for inference, including the boundary-aware attention modules, residual edge-aware graph convolution layers, encoder-decoder feature flow, and independent prediction heads for `Cp`, `U`, and `V`.

The training loop is not included in the public model-definition file.

## Normalization Parameters

The node input features and target variables were standardized using statistics calculated from the training set only.

The released normalization parameters are provided in:

```text
normalization/normalization_stats.json
normalization/scalers.pkl
```

The same training-set statistics are applied unchanged during validation, independent testing, and released-model inference.

## Single-Case Inference

The script

```text
reproducibility/inference.py
```

provides a runnable entry point for applying the released BA-GNN checkpoint to a single graph sample.

Example:

```bash
python reproducibility/inference.py \
    --data path/to/airfoil/dataset_pyg/example.pt \
    --checkpoint checkpoint/bagnn_seed0_best.pt \
    --normalization normalization/normalization_stats.json
```

The script loads the released checkpoint, applies the training-set normalization parameters, performs BA-GNN prediction, converts the outputs back to physical space, and saves the predicted `Cp`, `U`, and `V` values to a CSV file.

## Independent-Test Evaluation

The script

```text
reproducibility/evaluate.py
```

provides a runnable evaluation entry point for the exact independent test set defined in `fixed_split.json`.

Example:

```bash
python reproducibility/evaluate.py \
    --dataset_root path/to/full_graph_dataset \
    --split reproducibility/fixed_split.json \
    --checkpoints checkpoint/bagnn_seed0_best.pt \
    --normalization normalization/normalization_stats.json
```

The evaluation metrics are first calculated separately for each CFD case and are then aggregated across cases, consistent with the evaluation protocol described in the manuscript.

The script reports MAE, RMSE, and R² for `Cp`, `U`, and `V` over both the global flow field and the airfoil-boundary region.

The angle-of-attack cases are reported separately as:

- `1°–13°`: in-range unseen-geometry evaluation;
- `14°–20°`: preliminary out-of-range angle-of-attack evaluation.

## Fixed-Split Loader

The script

```text
reproducibility/fixed_split_loader.py
```

loads the exact train/validation/test airfoil lists used in the manuscript.

Rather than regenerating the data split from a random seed, this loader directly uses the released fixed split so that the same airfoil subsets can be recovered during independent evaluation.

## Model and Experimental Configuration

The file

```text
config.yaml
```

summarizes the principal model and experimental settings used in the study, including the BA-GNN architecture parameters, optimization settings, boundary-weight coefficient, and data-split information.

The released checkpoint also stores the principal architecture information required to reconstruct the corresponding BA-GNN model.

## Software Environment

The principal Python dependencies required for the released reproducibility scripts are listed in:

```text
requirements.txt
```

Additional environment information is provided in:

```text
environment_info.json
```

The experiments reported in the manuscript were conducted using PyTorch and PyTorch Geometric with CUDA acceleration.

## Scope of the Released Materials

The public materials are intended to support:

1. inspection of the dataset-construction procedure;
2. verification of the exact data split;
3. inspection of the BA-GNN model architecture;
4. reproduction of the preprocessing and normalization procedure;
5. inference using the released trained checkpoint;
6. independent recalculation of the reported test metrics.

The complete internal training implementation is not publicly released because it contains project-specific modules associated with ongoing research. However, the principal model architecture, experimental settings, preprocessing information, dataset, trained checkpoint, and runnable inference and evaluation procedures are provided to facilitate independent verification of the reported results.
