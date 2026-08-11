# BA-GNN Minimal Public Reproducibility Package

This package is designed to support transparent **inference and independent-test evaluation** of the BA-GNN model reported in the revised manuscript.

## What is included

- `reproducibility/model.py`: Full BA-GNN architecture required to load the released checkpoint.
- `reproducibility/inference.py`: Runnable single-case prediction entry point.
- `reproducibility/evaluate.py`: Runnable independent-test evaluation entry point. Metrics are calculated **per CFD case first and then aggregated**.
- `reproducibility/fixed_split_loader.py`: Loader for the exact geometry-level train/validation/test split used in the manuscript.
- `reproducibility/fixed_split.json`: Exact split: 338 training, 62 validation, and 8 independent-test airfoils.
- `reproducibility/test_airfoils.txt`: Names of the eight independent-test airfoils.
- `checkpoint/bagnn_seed0_best.pt`: Released trained Full BA-GNN checkpoint (seed 0, best epoch 437).
- `normalization/normalization_stats.json`: Training-set normalization statistics in an open JSON format.
- `normalization/scalers.pkl`: Original saved StandardScaler objects for audit/reference.
- `config.yaml`: Model, data-split, loss, and disclosed training settings.
- `environment_info.json` and `requirements.txt`: Environment information and dependencies.

The existing public dataset and dataset-construction code in the main BA-GNN repository should be retained alongside this package.

## Scope of this public package

The complete internal training implementation is **not included** because it belongs to an ongoing research project and contains project-specific modules subject to intellectual-property restrictions. The public package provides the model definition, exact data split, normalization parameters, trained checkpoint, and runnable inference/evaluation scripts so that users can independently load the reported model, perform predictions, and recalculate evaluation metrics on the released data.

The released checkpoint is a representative seed-0 model. If additional trained checkpoints for seeds 1 and 2 are released later, `evaluate.py` already supports multiple checkpoints and will report the corresponding across-checkpoint mean and standard deviation.

## Expected dataset layout

The scripts expect the released graph dataset to have the same layout used in the study:

```text
<dataset_root>/
  NACADAE11/
    dataset_pyg/
      NACADAE11_aoa_1.0.pt
      ...
  NACAEppler550/
    dataset_pyg/
      ...
  ...
```

Each PyTorch Geometric graph must contain:

- `x`: node features (4 columns)
- `y`: targets `[Cp, U, V]`
- `edge_index`
- `edge_attr` (5 columns)
- `node_type`
- `edge_type`

## Installation

Create a clean Python environment and install the dependencies:

```bash
pip install -r requirements.txt
```

The original experiments reported in the manuscript used PyTorch `2.0.0+cu118`, PyTorch Geometric `2.6.1`, CUDA `11.8`, and an NVIDIA RTX A6000. GPU users may need to install the appropriate PyTorch CUDA build for their system before installing the remaining dependencies.

## 1. Verify the exact fixed split

```bash
python reproducibility/fixed_split_loader.py \
  --split reproducibility/fixed_split.json \
  --dataset_root "PATH/TO/DATASET"
```

Expected airfoil counts:

```text
train:      338
validation: 62
test:       8
```

The independent test set contains 160 graph cases (20 angles for each of the eight unseen airfoils).

## 2. Run single-case inference

Example:

```bash
python reproducibility/inference.py \
  --data "PATH/TO/DATASET/NACADAE11/dataset_pyg/NACADAE11_aoa_17.0.pt" \
  --checkpoint checkpoint/bagnn_seed0_best.pt \
  --normalization normalization/normalization_stats.json \
  --output prediction_DAE11_17deg.csv
```

The output CSV contains nodal predictions of `Cp`, `U`, and `V` in physical space. If ground-truth targets are present in the graph file, they are also written to the CSV for direct comparison.

## 3. Recalculate independent-test metrics

```bash
python reproducibility/evaluate.py \
  --dataset_root "PATH/TO/DATASET" \
  --split reproducibility/fixed_split.json \
  --checkpoints checkpoint/bagnn_seed0_best.pt \
  --normalization normalization/normalization_stats.json \
  --out_dir evaluation_seed0
```

The script creates:

- `case_metrics.csv`: MAE, RMSE, and R² for each CFD case.
- `aggregate_by_checkpoint.csv`: case-mean metrics for the released checkpoint.
- `multi_checkpoint_summary.csv`: mean ± SD across all checkpoints passed to `--checkpoints`.

The two angle-of-attack groups are reported separately:

- `1-13_in_range`: unseen-geometry evaluation within the training angle range.
- `14-20_preliminary_out_of_range`: preliminary out-of-range angle-of-attack evaluation.

## Optional multi-checkpoint evaluation

If checkpoints from all three training seeds are available:

```bash
python reproducibility/evaluate.py \
  --dataset_root "PATH/TO/DATASET" \
  --checkpoints \
    checkpoint/bagnn_seed0_best.pt \
    checkpoint/bagnn_seed1_best.pt \
    checkpoint/bagnn_seed2_best.pt
```

This produces across-checkpoint mean and standard deviation without requiring access to the internal training implementation.

## Reproducibility note

The public materials are intended to enable independent verification of the released model and the reported evaluation protocol. The exact fixed split is loaded directly from `fixed_split.json`; it is not regenerated from a random seed during evaluation. Normalization uses statistics fitted only on the training set, as reported in the manuscript.
