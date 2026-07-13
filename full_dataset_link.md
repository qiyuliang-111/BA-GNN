# Full Dataset Access

The full BA-GNN graph dataset is available through Baidu Netdisk because its size is too large to be hosted directly in this GitHub repository.

## Download Information

- **Baidu Netdisk link:** `https://pan.baidu.com/s/1cbcrKv_f1c7VCudBYm8gHg`
- **Extraction code:** `aux6`
- **Approximate compressed size:** about 15 GB

## Dataset Contents

The full dataset contains graph samples generated from 408 airfoil geometries under multiple angles of attack.

- **Training set:** 338 airfoils, 4,394 graph samples, angles of attack from 1° to 13°
- **Validation set:** 62 airfoils, 806 graph samples, angles of attack from 1° to 13°
- **Independent test set:** 8 airfoils, 160 graph samples, angles of attack from 1° to 20°
- **Total:** 408 airfoils and 5,360 graph samples

Each graph sample is stored in PyTorch Geometric format and contains:

- node features: spatial coordinates, node type, and angle of attack;
- edge connectivity;
- edge features: relative coordinates, normalized edge length, and unit direction vector;
- node and edge boundary-type labels;
- target variables: pressure coefficient `Cp`, streamwise velocity `U`, and transverse velocity `V`.

The fixed geometry-level split used in the manuscript is provided in:

```text
dataset_split/data_split_table.csv
```

The dataset format and variable definitions are described in:

```text
dataset_description/data_format_description.md
```

A representative graph sample is also provided in the `example_data` directory.

## Notes

The dataset split was performed at the airfoil-geometry level. Therefore, all angle-of-attack cases corresponding to the same airfoil belong to the same subset, which prevents geometry-level data leakage.

The full dataset is provided for academic research and reproducibility purposes. Please cite the corresponding manuscript when using the dataset.

If the Baidu Netdisk link becomes unavailable or access problems occur, please contact the corresponding author.
