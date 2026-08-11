# -*- coding: utf-8 -*-
"""Runnable single-case inference entry point for the released BA-GNN checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data

from model import build_bagnn_from_checkpoint, safe_torch_load


def load_normalization(path: str):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    f_mean = np.asarray(obj["feature_mean"], dtype=np.float64)
    f_scale = np.asarray(obj["feature_scale"], dtype=np.float64)
    t_mean = np.asarray(obj["target_mean"], dtype=np.float64)
    t_scale = np.asarray(obj["target_scale"], dtype=np.float64)
    return f_mean, f_scale, t_mean, t_scale


def prepare_graph(g, f_mean, f_scale, device):
    required = ["x", "edge_index", "edge_attr", "node_type", "edge_type"]
    missing = [name for name in required if not hasattr(g, name) or getattr(g, name) is None]
    if missing:
        raise ValueError(f"Input graph is missing required fields: {missing}")

    x_raw = g.x.float().cpu().numpy()
    if x_raw.shape[1] != f_mean.size:
        raise ValueError(f"Input feature dimension mismatch: graph={x_raw.shape[1]}, normalization={f_mean.size}")
    x_norm = (x_raw - f_mean) / f_scale

    data = Data(
        x=torch.tensor(x_norm, dtype=torch.float32),
        edge_index=g.edge_index.long(),
        edge_attr=g.edge_attr.float() if g.edge_attr.dim() == 2 else g.edge_attr.float().unsqueeze(-1),
        node_type=g.node_type.long().view(-1),
        edge_type=g.edge_type.long().view(-1),
    ).to(device)
    return data, x_raw


def main():
    p = argparse.ArgumentParser(description="Run BA-GNN inference for one PyG graph case.")
    p.add_argument("--data", required=True, help="Path to one dataset_pyg/*.pt graph")
    p.add_argument("--checkpoint", default="checkpoint/bagnn_seed0_best.pt")
    p.add_argument("--normalization", default="normalization/normalization_stats.json")
    p.add_argument("--output", default=None, help="CSV output path; default: <input>_bagnn_prediction.csv")
    p.add_argument("--device", default=None, help="cuda, cpu, or omit for automatic selection")
    args = p.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = safe_torch_load(args.checkpoint, map_location=device)
    model = build_bagnn_from_checkpoint(checkpoint, device=device)
    f_mean, f_scale, t_mean, t_scale = load_normalization(args.normalization)

    g = safe_torch_load(args.data, map_location="cpu")
    data, x_raw = prepare_graph(g, f_mean, f_scale, device)

    with torch.no_grad():
        pred_z = model(data).detach().cpu().numpy()
    pred_phys = pred_z * t_scale.reshape(1, -1) + t_mean.reshape(1, -1)

    out_path = Path(args.output) if args.output else Path(args.data).with_suffix("").with_name(Path(args.data).stem + "_bagnn_prediction.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = None
    if hasattr(g, "y") and g.y is not None:
        y_true = g.y.float().cpu().numpy()
        if y_true.ndim == 1:
            y_true = y_true.reshape(-1, 1)
        if y_true.shape[1] > 3:
            y_true = y_true[:, :3]

    node_type = g.node_type.long().view(-1).cpu().numpy()
    with out_path.open("w", newline="", encoding="utf-8-sig") as f:
        fields = ["node_id", "x", "y", "node_type", "Cp_pred", "U_pred", "V_pred"]
        if y_true is not None and y_true.shape[1] >= 3:
            fields += ["Cp_true", "U_true", "V_true"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(pred_phys.shape[0]):
            row = {
                "node_id": i,
                "x": float(x_raw[i, 0]),
                "y": float(x_raw[i, 1]),
                "node_type": int(node_type[i]),
                "Cp_pred": float(pred_phys[i, 0]),
                "U_pred": float(pred_phys[i, 1]),
                "V_pred": float(pred_phys[i, 2]),
            }
            if y_true is not None and y_true.shape[1] >= 3:
                row.update({"Cp_true": float(y_true[i, 0]), "U_true": float(y_true[i, 1]), "V_true": float(y_true[i, 2])})
            writer.writerow(row)

    print(f"Device: {device}")
    print(f"Checkpoint seed: {checkpoint.get('seed', 'unknown')}")
    print(f"Best epoch: {checkpoint.get('best_epoch', 'unknown')}")
    print(f"Prediction shape: {pred_phys.shape}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
