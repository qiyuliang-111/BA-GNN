# -*- coding: utf-8 -*-
"""Exact fixed-split loader for the BA-GNN reproducibility package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

REQUIRED_KEYS = {"train_airfoils", "val_airfoils", "test_airfoils"}


def load_fixed_split(path: str) -> Dict:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(split)
    if missing:
        raise ValueError(f"Missing split keys: {sorted(missing)}")

    train = set(split["train_airfoils"])
    val = set(split["val_airfoils"])
    test = set(split["test_airfoils"])
    if train & val or train & test or val & test:
        raise ValueError("The fixed train/validation/test airfoil sets overlap.")

    expected = {
        "train": split.get("num_airfoils_train"),
        "val": split.get("num_airfoils_val"),
        "test": split.get("num_airfoils_test"),
    }
    actual = {"train": len(train), "val": len(val), "test": len(test)}
    for key in actual:
        if expected[key] is not None and int(expected[key]) != actual[key]:
            raise ValueError(f"{key} split count mismatch: expected {expected[key]}, got {actual[key]}")
    return split


def collect_graph_files(dataset_root: str, split_file: str, subset: str = "test") -> List[str]:
    if subset not in {"train", "val", "test"}:
        raise ValueError("subset must be one of: train, val, test")
    split = load_fixed_split(split_file)
    airfoils = split[f"{subset}_airfoils"]
    root = Path(dataset_root)
    files: List[str] = []
    missing_dirs = []

    for airfoil in airfoils:
        d = root / airfoil / "dataset_pyg"
        if not d.is_dir():
            missing_dirs.append(str(d))
            continue
        files.extend(str(p) for p in sorted(d.glob("*.pt")))

    if missing_dirs:
        preview = "\n".join(missing_dirs[:10])
        raise FileNotFoundError(
            f"Missing dataset_pyg directories for {len(missing_dirs)} airfoils. First entries:\n{preview}"
        )
    return files


def main():
    p = argparse.ArgumentParser(description="Validate and inspect the exact fixed airfoil split.")
    p.add_argument("--split", default="reproducibility/fixed_split.json")
    p.add_argument("--dataset_root", default=None)
    args = p.parse_args()

    split = load_fixed_split(args.split)
    print("split_type:", split.get("split_type"))
    print("split_seed:", split.get("split_seed"))
    print("train airfoils:", len(split["train_airfoils"]))
    print("validation airfoils:", len(split["val_airfoils"]))
    print("test airfoils:", len(split["test_airfoils"]))

    if args.dataset_root:
        for subset in ("train", "val", "test"):
            files = collect_graph_files(args.dataset_root, args.split, subset)
            print(f"{subset} graph files: {len(files)}")


if __name__ == "__main__":
    main()
