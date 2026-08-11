# -*- coding: utf-8 -*-
"""Independent-test evaluation entry point for released BA-GNN checkpoint(s).

Metrics are calculated per CFD case first and then aggregated across cases,
matching the evaluation protocol described in the revised manuscript.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch_geometric.data import Data

from fixed_split_loader import collect_graph_files, load_fixed_split
from model import build_bagnn_from_checkpoint, safe_torch_load

VAR_NAMES = ["Cp", "U", "V"]
NODE_TYPE_AIRFOIL = 1


def load_normalization(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        np.asarray(obj["feature_mean"], dtype=np.float64),
        np.asarray(obj["feature_scale"], dtype=np.float64),
        np.asarray(obj["target_mean"], dtype=np.float64),
        np.asarray(obj["target_scale"], dtype=np.float64),
    )


def metrics_np(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    diff = y_pred - y_true
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def aoa_group(aoa: float) -> str:
    if 1.0 <= aoa <= 13.0:
        return "1-13_in_range"
    if 14.0 <= aoa <= 20.0:
        return "14-20_preliminary_out_of_range"
    return "other"


def prepare_graph(g, f_mean, f_scale, device):
    required = ["x", "y", "edge_index", "edge_attr", "node_type", "edge_type"]
    missing = [name for name in required if not hasattr(g, name) or getattr(g, name) is None]
    if missing:
        raise ValueError(f"Graph is missing required fields: {missing}")

    x_raw = g.x.float().cpu().numpy()
    y_true = g.y.float().cpu().numpy()
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_true.shape[1] > 3:
        y_true = y_true[:, :3]
    x_norm = (x_raw - f_mean) / f_scale

    data = Data(
        x=torch.tensor(x_norm, dtype=torch.float32),
        edge_index=g.edge_index.long(),
        edge_attr=g.edge_attr.float() if g.edge_attr.dim() == 2 else g.edge_attr.float().unsqueeze(-1),
        node_type=g.node_type.long().view(-1),
        edge_type=g.edge_type.long().view(-1),
    ).to(device)
    return data, x_raw, y_true, g.node_type.long().view(-1).cpu().numpy()


def write_csv(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def nanmean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")


def nanstd(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(np.std(arr, ddof=1)) if arr.size > 1 else (0.0 if arr.size == 1 else float("nan"))


def evaluate_checkpoint(checkpoint_path: str, graph_files: List[str], normalization_path: str, device) -> List[Dict]:
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)
    model = build_bagnn_from_checkpoint(checkpoint, device=device)
    f_mean, f_scale, t_mean, t_scale = load_normalization(normalization_path)
    seed = checkpoint.get("seed", "unknown")

    rows: List[Dict] = []
    for fp in graph_files:
        g = safe_torch_load(fp, map_location="cpu")
        data, x_raw, y_true, node_type = prepare_graph(g, f_mean, f_scale, device)
        with torch.no_grad():
            pred_z = model(data).detach().cpu().numpy()
        pred = pred_z * t_scale.reshape(1, -1) + t_mean.reshape(1, -1)

        aoa = float(np.mean(x_raw[:, -1]))
        airfoil = Path(fp).parent.parent.name
        case_name = Path(fp).name
        masks = {
            "global": np.ones(node_type.shape[0], dtype=bool),
            "airfoil": node_type == NODE_TYPE_AIRFOIL,
        }
        for region, mask in masks.items():
            if not np.any(mask):
                continue
            for j, var in enumerate(VAR_NAMES):
                m = metrics_np(y_true[mask, j], pred[mask, j])
                rows.append({
                    "checkpoint": Path(checkpoint_path).name,
                    "seed": seed,
                    "airfoil": airfoil,
                    "case": case_name,
                    "aoa": aoa,
                    "aoa_group": aoa_group(aoa),
                    "region": region,
                    "variable": var,
                    **m,
                })
    return rows


def aggregate_case_rows(rows: List[Dict]) -> List[Dict]:
    keys = sorted({(r["checkpoint"], r["seed"], r["aoa_group"], r["region"], r["variable"]) for r in rows})
    out = []
    for checkpoint, seed, group, region, variable in keys:
        sub = [r for r in rows if (r["checkpoint"], r["seed"], r["aoa_group"], r["region"], r["variable"]) == (checkpoint, seed, group, region, variable)]
        out.append({
            "checkpoint": checkpoint,
            "seed": seed,
            "aoa_group": group,
            "region": region,
            "variable": variable,
            "num_cases": len(sub),
            "MAE_case_mean": nanmean(r["MAE"] for r in sub),
            "RMSE_case_mean": nanmean(r["RMSE"] for r in sub),
            "R2_case_mean": nanmean(r["R2"] for r in sub),
        })
    return out


def summarize_checkpoints(agg_rows: List[Dict]) -> List[Dict]:
    keys = sorted({(r["aoa_group"], r["region"], r["variable"]) for r in agg_rows})
    out = []
    for group, region, variable in keys:
        sub = [r for r in agg_rows if (r["aoa_group"], r["region"], r["variable"]) == (group, region, variable)]
        out.append({
            "aoa_group": group,
            "region": region,
            "variable": variable,
            "num_checkpoints": len(sub),
            "MAE_mean": nanmean(r["MAE_case_mean"] for r in sub),
            "MAE_std": nanstd(r["MAE_case_mean"] for r in sub),
            "RMSE_mean": nanmean(r["RMSE_case_mean"] for r in sub),
            "RMSE_std": nanstd(r["RMSE_case_mean"] for r in sub),
            "R2_mean": nanmean(r["R2_case_mean"] for r in sub),
            "R2_std": nanstd(r["R2_case_mean"] for r in sub),
        })
    return out


def main():
    p = argparse.ArgumentParser(description="Evaluate BA-GNN checkpoint(s) on the exact independent test split.")
    p.add_argument("--dataset_root", required=True, help="Root containing <airfoil>/dataset_pyg/*.pt")
    p.add_argument("--split", default="reproducibility/fixed_split.json")
    p.add_argument("--checkpoints", nargs="+", default=["checkpoint/bagnn_seed0_best.pt"])
    p.add_argument("--normalization", default="normalization/normalization_stats.json")
    p.add_argument("--out_dir", default="evaluation_output")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    split = load_fixed_split(args.split)
    graph_files = collect_graph_files(args.dataset_root, args.split, subset="test")

    expected_cases = split.get("num_cases_test")
    if expected_cases is not None and len(graph_files) != int(expected_cases):
        print(f"WARNING: fixed split expects {expected_cases} test cases, but {len(graph_files)} .pt files were found.")

    all_case_rows: List[Dict] = []
    for ckpt in args.checkpoints:
        print(f"Evaluating {ckpt} on {len(graph_files)} test cases ...")
        all_case_rows.extend(evaluate_checkpoint(ckpt, graph_files, args.normalization, device))

    agg_rows = aggregate_case_rows(all_case_rows)
    summary_rows = summarize_checkpoints(agg_rows)

    out_dir = Path(args.out_dir)
    write_csv(out_dir / "case_metrics.csv", all_case_rows)
    write_csv(out_dir / "aggregate_by_checkpoint.csv", agg_rows)
    write_csv(out_dir / "multi_checkpoint_summary.csv", summary_rows)

    print(f"Device: {device}")
    print(f"Fixed independent test airfoils: {len(split['test_airfoils'])}")
    print(f"Test graph cases found: {len(graph_files)}")
    print(f"Checkpoints evaluated: {len(args.checkpoints)}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
