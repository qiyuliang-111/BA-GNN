# BA-GNN Supplementary and Reproducibility Materials

This repository provides supplementary and reproducibility materials for the manuscript:

**A Novel Boundary-Aware Graph Neural Network with Weighted Loss for Airfoil Flow-Field Prediction**

The released materials are intended to support transparent inspection of the dataset construction, exact data split, model architecture, preprocessing procedure, Full BA-GNN training protocol, trained-model inference, and independent-test evaluation reported in the manuscript.

## Repository Contents

This repository provides:

* the exact geometry-level train/validation/test split used in the manuscript;
* the dataset-construction code;
* a detailed description of the graph-data format;
* access information for the complete graph dataset;
* the Full BA-GNN model definition;
* a runnable minimal training entry point for Full BA-GNN;
* the best-performing BA-GNN checkpoints for random seeds **0, 1, and 2**;
* the corresponding training and evaluation results for the three seeds;
* normalization parameters derived from the training set;
* runnable single-case inference code;
* runnable independent-test evaluation code;
* model and experimental configuration information;
* software-environment and dependency information.

The public training script is a minimal standalone implementation of the disclosed Full BA-GNN training protocol. The complete internal research codebase, including project-specific experiment-management modules, is not released.

---

## Repository Structure

```text
BA-GNN/
├── checkpoint/
│   ├── bagnn_seed0_best.pt
│   ├── bagnn_seed1_best.pt
│   └── bagnn_seed2_best.pt
│
├── results/
│   ├── seed0/
│   │   ├── training_history.csv
│   │   ├── aggregate_metrics.csv
│   │   └── run_summary.json
│   ├── seed1/
│   │   ├── training_history.csv
│   │   ├── aggregate_metrics.csv
│   │   └── run_summary.json
│   ├── seed2/
│   │   ├── training_history.csv
│   │   ├── aggregate_metrics.csv
│   │   └── run_summary.json
│   └── three_seed_summary.csv
│
├── normalization/
│   ├── normalization_stats.json
│   └── scalers.pkl
│
├── reproducibility/
│   ├── model.py
│   ├── train.py
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

---

## Dataset Split

A **geometry-level data split** is used to prevent information leakage between different angle-of-attack cases belonging to the same airfoil.

The complete dataset contains **408 airfoil geometries**:

* **338 airfoils** for training;
* **62 airfoils** for validation;
* **8 unseen airfoils** reserved for independent testing.

All angle-of-attack cases associated with the same airfoil are assigned to the same subset.

The exact split used in the manuscript is provided in:

```text
reproducibility/fixed_split.json
```

The eight independent test airfoils are additionally listed in:

```text
reproducibility/test_airfoils.txt
```

Case-level split information is provided in:

```text
dataset_split/data_split_table.csv
```

For data-processing convenience, the prefix `NACA` was added to some internal airfoil filenames. This prefix is only an internal filename convention and does not necessarily indicate the actual airfoil family.

---

## Dataset Construction

The graph-dataset construction procedure is provided in:

```text
dataset_creation_code/dataset_generation.py
```

The main processing steps include:

* CFD mesh-to-graph conversion;
* automatic boundary-type identification;
* node-feature construction;
* edge-feature construction;
* graph-connectivity construction;
* preprocessing and serialization into PyTorch Geometric graph samples.

---

## Graph Data Format

Each flow-field case is stored as a PyTorch Geometric graph object.

The graph samples contain the information required by BA-GNN, including:

* node features;
* graph connectivity (`edge_index`);
* continuous edge attributes (`edge_attr`);
* node-type labels (`node_type`);
* edge-type labels (`edge_type`);
* target flow-field variables.

The prediction targets are:

* pressure coefficient, `Cp`;
* streamwise velocity component, `U`;
* transverse velocity component, `V`.

A detailed description is available in:

```text
dataset_description/data_format_description.md
```

---

## Full Dataset

Because of the size of the complete graph dataset, it is distributed through the external access information provided in:

```text
full_dataset_link.md
```

The complete dataset can be used together with the fixed split, normalization parameters, and reproducibility scripts provided in this repository.

---

## Normalization

Node input features and prediction targets are standardized using statistics calculated from the **training set only**.

The released normalization information is provided in:

```text
normalization/normalization_stats.json
normalization/scalers.pkl
```

The same training-set statistics are applied unchanged to the validation and independent-test data.

Continuous edge features are constructed in normalized form during graph generation and are not further Z-score standardized.

---

## Full BA-GNN Model

The public Full BA-GNN architecture is implemented in:

```text
reproducibility/model.py
```

The model includes:

* boundary-aware multi-head attention;
* continuous edge-feature encoding;
* boundary-semantic information;
* residual edge-aware graph convolution;
* feature-level encoder-decoder skip connections;
* independent prediction heads for `Cp`, `U`, and `V`.

---

## Released Three-Seed Checkpoints

The best-performing Full BA-GNN checkpoint from each of the three independent training runs is publicly provided:

```text
checkpoint/bagnn_seed0_best.pt
checkpoint/bagnn_seed1_best.pt
checkpoint/bagnn_seed2_best.pt
```

These checkpoints correspond to the three random seeds used for the statistical results reported in the manuscript:

```text
0, 1, 2
```

Together with the fixed split, released normalization parameters, and evaluation script, these checkpoints enable independent verification of the reported **three-seed mean ± SD evaluation statistics**.

---

## Training Protocol

A runnable minimal Full BA-GNN training entry point is provided in:

```text
reproducibility/train.py
```

The public training script implements the disclosed training protocol used for Full BA-GNN, including:

* the fixed geometry-level training and validation split;
* the released training-set normalization parameters;
* boundary-weighted multi-task MSE;
* Adam optimization;
* learning-rate scheduling;
* gradient clipping;
* validation-based checkpoint selection;
* early stopping;
* random-seed control.

The principal training settings are summarized in:

```text
config.yaml
```

The reported Full BA-GNN training protocol uses:

| Setting                     |             Value |
| --------------------------- | ----------------: |
| Random seeds                |           0, 1, 2 |
| Batch size                  |                 8 |
| Maximum epochs              |               500 |
| Optimizer                   |              Adam |
| Initial learning rate       |          1 × 10⁻⁴ |
| Weight decay                |          1 × 10⁻⁵ |
| LR scheduler                | ReduceLROnPlateau |
| Scheduler factor            |               0.5 |
| Scheduler patience          |         20 epochs |
| Minimum learning rate       |          1 × 10⁻⁶ |
| Gradient clipping           |               1.0 |
| Early-stopping patience     |        100 epochs |
| Boundary-weight coefficient |            β = 10 |
| Loss normalization          |              sumW |

Example training command:

```bash
python reproducibility/train.py \
    --dataset_root "PATH/TO/FULL_GRAPH_DATASET" \
    --seed 0 \
    --output_dir "training_output/seed0"
```

For the other two manuscript seeds, replace `--seed 0` with:

```text
--seed 1
```

or:

```text
--seed 2
```

The public script outputs a best checkpoint, training history, and run summary.

The public `train.py` is intended as a minimal standalone implementation of the disclosed Full BA-GNN training protocol. It does not contain the complete internal experiment-management framework used during the broader research workflow.

---

## Released Training and Evaluation Results

Seed-specific training and evaluation outputs are provided in:

```text
results/seed0/
results/seed1/
results/seed2/
```

For each random seed, the released files include:

### `training_history.csv`

Epoch-wise training information, including training loss, validation loss, and learning-rate evolution.

### `aggregate_metrics.csv`

Aggregated evaluation metrics for the corresponding trained model.

### `run_summary.json`

Summary information for the training run, including the random seed, best epoch, validation loss, parameter count, training time, and inference information.

The across-seed statistical summary is provided in:

```text
results/three_seed_summary.csv
```

This file reports the results aggregated across seeds 0, 1, and 2 and provides the statistics required to verify the reported three-seed mean ± SD values.

---

## Single-Case Inference

A runnable single-case inference entry point is provided in:

```text
reproducibility/inference.py
```

Example:

```bash
python reproducibility/inference.py \
    --data "PATH/TO/GRAPH_SAMPLE.pt" \
    --checkpoint checkpoint/bagnn_seed0_best.pt \
    --normalization normalization/normalization_stats.json
```

The script:

1. loads a graph sample;
2. applies the released training-set normalization parameters;
3. loads the selected Full BA-GNN checkpoint;
4. performs inference;
5. converts predictions back to physical space;
6. exports predicted `Cp`, `U`, and `V`.

Any of the three released checkpoints can be used by changing the checkpoint path.

---

## Independent-Test Evaluation

The independent-test evaluation entry point is provided in:

```text
reproducibility/evaluate.py
```

The script uses the exact independent-airfoil split stored in:

```text
reproducibility/fixed_split.json
```

### Single-checkpoint evaluation

```bash
python reproducibility/evaluate.py \
    --dataset_root "PATH/TO/FULL_GRAPH_DATASET" \
    --split reproducibility/fixed_split.json \
    --checkpoints checkpoint/bagnn_seed0_best.pt \
    --normalization normalization/normalization_stats.json
```

### Three-checkpoint evaluation

```bash
python reproducibility/evaluate.py \
    --dataset_root "PATH/TO/FULL_GRAPH_DATASET" \
    --split reproducibility/fixed_split.json \
    --checkpoints \
        checkpoint/bagnn_seed0_best.pt \
        checkpoint/bagnn_seed1_best.pt \
        checkpoint/bagnn_seed2_best.pt \
    --normalization normalization/normalization_stats.json
```

Evaluation metrics are first calculated separately for each CFD case and are then aggregated across cases, consistent with the statistical protocol described in the manuscript.

The reported metrics include:

* MAE;
* RMSE;
* R²;

for:

* `Cp`;
* `U`;
* `V`.

The evaluation distinguishes between:

* the global flow field;
* the airfoil-boundary region.

For the independent test airfoils, angle-of-attack cases are additionally distinguished as:

* **1°–13°:** in-range unseen-geometry evaluation;
* **14°–20°:** preliminary out-of-range angle-of-attack evaluation.

When all three released checkpoints are supplied, the evaluation procedure can be used to independently verify the corresponding three-seed statistics.

---

## Software Environment

The principal Python dependencies are listed in:

```text
requirements.txt
```

Additional environment information is provided in:

```text
environment_info.json
```

The experiments reported in the manuscript were conducted using PyTorch and PyTorch Geometric with CUDA acceleration.

---

## Scope of the Released Materials

The public materials are intended to support:

1. inspection of the graph-dataset construction procedure;
2. verification of the exact geometry-level data split;
3. inspection of the Full BA-GNN architecture;
4. reproduction of preprocessing and normalization;
5. execution of a minimal Full BA-GNN training workflow;
6. inference using the three released trained checkpoints;
7. independent recalculation of the reported test metrics;
8. independent verification of the reported three-seed mean ± SD evaluation statistics.

The repository does not contain the complete internal research codebase or project-specific experiment-management framework. These components are not required for using the released model architecture, minimal public training entry point, trained checkpoints, or evaluation workflow.

---

## Manuscript

**A Novel Boundary-Aware Graph Neural Network with Weighted Loss for Airfoil Flow-Field Prediction**

Additional citation information will be added after publication.
