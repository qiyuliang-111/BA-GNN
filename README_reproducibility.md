# BA-GNN Public Reproducibility Package

This document provides a concise guide to the public reproducibility materials for the manuscript:

**A Novel Boundary-Aware Graph Neural Network with Weighted Loss for Airfoil Flow-Field Prediction**

For a complete description of the repository, please refer to the main `README.md`.

## 1. Included Reproducibility Materials

The public package contains:

* `reproducibility/model.py`
  Full BA-GNN model architecture.

* `reproducibility/train.py`
  Runnable minimal training entry point for Full BA-GNN.

* `reproducibility/inference.py`
  Runnable single-case inference entry point.

* `reproducibility/evaluate.py`
  Runnable independent-test evaluation entry point. Metrics are first calculated for each CFD case and then aggregated.

* `reproducibility/fixed_split_loader.py`
  Loader for the exact geometry-level train/validation/test split.

* `reproducibility/fixed_split.json`
  Exact split containing 338 training, 62 validation, and 8 independent-test airfoils.

* `reproducibility/test_airfoils.txt`
  Names of the eight independent-test airfoils.

* `checkpoint/bagnn_seed0_best.pt`
  Best-performing Full BA-GNN checkpoint for random seed 0.

* `checkpoint/bagnn_seed1_best.pt`
  Best-performing Full BA-GNN checkpoint for random seed 1.

* `checkpoint/bagnn_seed2_best.pt`
  Best-performing Full BA-GNN checkpoint for random seed 2.

* `results/seed0/`, `results/seed1/`, and `results/seed2/`
  Seed-specific training histories, aggregated evaluation metrics, and run summaries.

* `results/three_seed_summary.csv`
  Across-seed statistical summary for seeds 0, 1, and 2.

* `normalization/normalization_stats.json`
  Training-set normalization parameters in JSON format.

* `normalization/scalers.pkl`
  Saved StandardScaler objects for audit/reference.

* `config.yaml`
  Model architecture, loss settings, random seeds, and disclosed training configurations.

* `environment_info.json` and `requirements.txt`
  Software environment and dependency information.

The dataset-construction code, dataset description, exact split information, representative graph sample, and full-dataset access information are also available in the main repository.

## 2. Scope of the Public Training Implementation

The public `reproducibility/train.py` is a minimal standalone implementation of the disclosed Full BA-GNN training protocol.

It includes:

* fixed geometry-level training and validation splits;
* released training-set normalization parameters;
* boundary-weighted multi-task MSE;
* Adam optimization;
* `ReduceLROnPlateau` learning-rate scheduling;
* gradient clipping;
* validation-based best-checkpoint selection;
* early stopping;
* random-seed control.

The complete internal research codebase and project-specific experiment-management framework are not included.

The released minimal training entry point, three trained checkpoints, fixed split, normalization parameters, and evaluation scripts are sufficient to inspect the disclosed training procedure and independently verify the reported three-seed evaluation statistics.

## 3. Expected Dataset Layout

The scripts expect the graph dataset to follow the structure:

```text
<dataset_root>/
├── Airfoil_A/
│   └── dataset_pyg/
│       ├── Airfoil_A_aoa_1.0.pt
│       ├── Airfoil_A_aoa_2.0.pt
│       └── ...
├── Airfoil_B/
│   └── dataset_pyg/
│       └── ...
└── ...
```

Each PyTorch Geometric graph contains:

* `x`: node features;
* `y`: targets `[Cp, U, V]`;
* `edge_index`;
* `edge_attr`;
* `node_type`;
* `edge_type`.

## 4. Installation

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

The experiments reported in the manuscript used:

* PyTorch `2.0.0+cu118`;
* PyTorch Geometric `2.6.1`;
* CUDA `11.8`;
* NVIDIA RTX A6000 GPU.

GPU users may need to install a PyTorch CUDA build appropriate for their local system.

## 5. Verify the Fixed Geometry-Level Split

Run:

```bash
python reproducibility/fixed_split_loader.py \
  --split reproducibility/fixed_split.json \
  --dataset_root "PATH/TO/DATASET"
```

Expected airfoil counts:

```text
train:       338
validation:   62
test:          8
```

The independent test set contains 160 graph cases corresponding to 20 angle-of-attack conditions for each of the eight unseen airfoils.

## 6. Minimal Full BA-GNN Training

Example for random seed 0:

```bash
python reproducibility/train.py \
  --dataset_root "PATH/TO/DATASET" \
  --seed 0 \
  --output_dir "training_output/seed0"
```

For the other two manuscript training seeds, replace `--seed 0` with:

```text
--seed 1
```

or:

```text
--seed 2
```

The disclosed training settings include:

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

The public training entry point outputs the best checkpoint, training history, and run summary.

## 7. Released Three-Seed Checkpoints

The best-performing Full BA-GNN checkpoints for the three random seeds used in the manuscript are:

```text
checkpoint/bagnn_seed0_best.pt
checkpoint/bagnn_seed1_best.pt
checkpoint/bagnn_seed2_best.pt
```

These checkpoints can be used directly with the released inference and evaluation scripts.

## 8. Single-Case Inference

Example:

```bash
python reproducibility/inference.py \
  --data "PATH/TO/GRAPH_SAMPLE.pt" \
  --checkpoint checkpoint/bagnn_seed0_best.pt \
  --normalization normalization/normalization_stats.json \
  --output prediction.csv
```

The output contains nodal predictions of `Cp`, `U`, and `V` in physical space.

Any of the three released checkpoints can be used by changing the checkpoint path.

## 9. Independent-Test Evaluation

### Single-checkpoint evaluation

```bash
python reproducibility/evaluate.py \
  --dataset_root "PATH/TO/DATASET" \
  --split reproducibility/fixed_split.json \
  --checkpoints checkpoint/bagnn_seed0_best.pt \
  --normalization normalization/normalization_stats.json \
  --out_dir evaluation_seed0
```

### Three-checkpoint evaluation

```bash
python reproducibility/evaluate.py \
  --dataset_root "PATH/TO/DATASET" \
  --split reproducibility/fixed_split.json \
  --checkpoints \
    checkpoint/bagnn_seed0_best.pt \
    checkpoint/bagnn_seed1_best.pt \
    checkpoint/bagnn_seed2_best.pt \
  --normalization normalization/normalization_stats.json \
  --out_dir evaluation_three_seeds
```

The evaluation script reports:

* case-level MAE, RMSE, and R²;
* checkpoint-level aggregated metrics;
* across-checkpoint mean and standard deviation.

The independent-test evaluation distinguishes:

* `1-13_in_range`: unseen-airfoil evaluation within the training angle-of-attack range;
* `14-20_preliminary_out_of_range`: preliminary evaluation outside the training angle-of-attack range.

## 10. Released Seed-Specific Results

The original outputs corresponding to the three Full BA-GNN training runs are provided in:

```text
results/
├── seed0/
│   ├── training_history.csv
│   ├── aggregate_metrics.csv
│   └── run_summary.json
├── seed1/
│   ├── training_history.csv
│   ├── aggregate_metrics.csv
│   └── run_summary.json
├── seed2/
│   ├── training_history.csv
│   ├── aggregate_metrics.csv
│   └── run_summary.json
└── three_seed_summary.csv
```

`training_history.csv` records the epoch-wise training process.

`aggregate_metrics.csv` provides the aggregated evaluation results for the corresponding trained checkpoint.

`run_summary.json` records the principal information for each training run.

`three_seed_summary.csv` provides the across-seed statistical summary used to verify the reported three-seed mean ± SD results.

## 11. Reproducibility Note

The exact geometry-level split is loaded directly from `fixed_split.json` and is not regenerated during evaluation.

Normalization uses parameters fitted only on the training set and subsequently applied unchanged to the validation and independent-test data.

The three released checkpoints, corresponding seed-specific results, fixed split, normalization parameters, and runnable evaluation scripts enable independent verification of the reported three-seed evaluation statistics.
