# -*- coding: utf-8 -*-
"""
Minimal public training entry point for Full BA-GNN.

This script is intentionally limited to the training protocol needed to
reproduce the Full BA-GNN runs reported in the revised manuscript. It reuses
the public model definition, exact geometry-level split, released
normalization parameters, and config.yaml.

Example
-------
python reproducibility/train.py \
  --dataset_root "D:/SA-GNN/dataset" \
  --seed 0 \
  --output_dir "training_output/seed0"

Repeat with --seed 1 and --seed 2 for the three reported training seeds.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from fixed_split_loader import collect_graph_files
from model import BAGNN, safe_torch_load


NODE_TYPE_AIRFOIL = 1


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python/NumPy/PyTorch random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        # Matches the original training workflow.
        torch.backends.cudnn.benchmark = True


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_normalization(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    return (
        np.asarray(obj["feature_mean"], dtype=np.float64),
        np.asarray(obj["feature_scale"], dtype=np.float64),
        np.asarray(obj["target_mean"], dtype=np.float64),
        np.asarray(obj["target_scale"], dtype=np.float64),
    )


class GraphFileDataset(Dataset):
    """
    Lazy loader for PyG graph files.

    Input features and Cp/U/V targets are standardized with the released
    training-set statistics. Edge features are kept as stored in the public
    graph files because they are already constructed in normalized form.
    """

    def __init__(
        self,
        files: List[str],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
        target_mean: np.ndarray,
        target_scale: np.ndarray,
    ):
        self.files = list(files)
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.target_mean = target_mean
        self.target_scale = target_scale

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Data:
        fp = self.files[idx]
        g = safe_torch_load(fp, map_location="cpu")

        required = ["x", "y", "edge_index", "edge_attr", "node_type", "edge_type"]
        missing = [name for name in required if not hasattr(g, name) or getattr(g, name) is None]
        if missing:
            raise ValueError(f"{fp} is missing required fields: {missing}")

        x_raw = g.x.float().cpu().numpy()
        y_raw = g.y.float().cpu().numpy()

        if y_raw.ndim == 1:
            y_raw = y_raw.reshape(-1, 1)
        if y_raw.shape[1] < 3:
            raise ValueError(f"{fp}: expected at least 3 target columns, got {y_raw.shape[1]}")
        if y_raw.shape[1] > 3:
            y_raw = y_raw[:, :3]

        if x_raw.shape[1] != self.feature_mean.size:
            raise ValueError(
                f"{fp}: input dimension {x_raw.shape[1]} does not match "
                f"normalization dimension {self.feature_mean.size}"
            )
        if y_raw.shape[1] != self.target_mean.size:
            raise ValueError(
                f"{fp}: target dimension {y_raw.shape[1]} does not match "
                f"normalization dimension {self.target_mean.size}"
            )

        x_norm = (x_raw - self.feature_mean) / self.feature_scale
        y_norm = (y_raw - self.target_mean) / self.target_scale

        edge_attr = g.edge_attr.float()
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)

        return Data(
            x=torch.tensor(x_norm, dtype=torch.float32),
            y=torch.tensor(y_norm, dtype=torch.float32),
            edge_index=g.edge_index.long(),
            edge_attr=edge_attr,
            node_type=g.node_type.long().view(-1),
            edge_type=g.edge_type.long().view(-1),
        )


class BoundaryWeightedMSE(nn.Module):
    """
    Boundary-weighted multi-task MSE used for Full BA-GNN.

    For sumW normalization:
        L = sum_i w_i * sum_m q_m * e_{i,m}^2
            / (sum_i w_i * sum_m q_m)

    With task_weights=[1,1,1], this is equivalent to the revised manuscript
    formulation using beta=10 for airfoil-surface nodes.
    """

    def __init__(
        self,
        beta: float = 10.0,
        task_weights: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        normalization: str = "sumW",
    ):
        super().__init__()
        self.beta = float(beta)
        self.normalization = str(normalization)
        self.register_buffer(
            "task_weights",
            torch.tensor(task_weights, dtype=torch.float32).view(1, -1),
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        sq = (pred - target) ** 2
        task_w = self.task_weights.to(dtype=sq.dtype, device=sq.device)

        node_w = torch.ones((sq.size(0), 1), dtype=sq.dtype, device=sq.device)
        node_w[node_type == NODE_TYPE_AIRFOIL] = self.beta

        weighted_sq = sq * node_w * task_w

        if self.normalization.lower() == "sumw":
            denom = node_w.sum() * task_w.sum()
            return weighted_sq.sum() / denom.clamp_min(1.0)

        if self.normalization.lower() == "n":
            return weighted_sq.mean()

        raise ValueError("loss normalization must be 'sumW' or 'N'")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip: float,
) -> float:
    model.train()
    loss_sum = 0.0
    node_sum = 0

    for batch in loader:
        batch = batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(batch)
        loss = criterion(pred, batch.y, batch.node_type)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        n = int(batch.num_nodes)
        loss_sum += float(loss.item()) * n
        node_sum += n

    return loss_sum / max(node_sum, 1)


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    loss_sum = 0.0
    node_sum = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch)
        loss = criterion(pred, batch.y, batch.node_type)

        n = int(batch.num_nodes)
        loss_sum += float(loss.item()) * n
        node_sum += n

    return loss_sum / max(node_sum, 1)


def write_history(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train Full BA-GNN on the exact public geometry-level split."
    )
    p.add_argument("--dataset_root", required=True, help="Root containing <airfoil>/dataset_pyg/*.pt")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--split", default="reproducibility/fixed_split.json")
    p.add_argument("--normalization", default="normalization/normalization_stats.json")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--device", default=None, help="e.g. cuda, cuda:0, or cpu")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    model_cfg = cfg["model"]
    train_cfg = cfg["training_protocol_disclosed"]
    loss_cfg = cfg["loss"]

    disclosed_seeds = list(train_cfg.get("training_seeds", [0, 1, 2]))
    if args.seed not in disclosed_seeds:
        print(
            f"WARNING: seed {args.seed} is not one of the manuscript seeds "
            f"{disclosed_seeds}. Training will continue."
        )

    output_dir = Path(args.output_dir or f"training_output/seed{args.seed}")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    set_seed(args.seed, deterministic=args.deterministic)

    feature_mean, feature_scale, target_mean, target_scale = load_normalization(
        args.normalization
    )

    train_files = collect_graph_files(args.dataset_root, args.split, subset="train")
    val_files = collect_graph_files(args.dataset_root, args.split, subset="val")

    if not train_files:
        raise RuntimeError("No training graph files were found.")
    if not val_files:
        raise RuntimeError("No validation graph files were found.")

    print("=" * 78)
    print("Full BA-GNN public training entry point")
    print("=" * 78)
    print(f"device       : {device}")
    print(f"seed         : {args.seed}")
    print(f"train cases  : {len(train_files)}")
    print(f"val cases    : {len(val_files)}")
    print(f"output dir   : {output_dir}")
    print("=" * 78)

    train_dataset = GraphFileDataset(
        train_files, feature_mean, feature_scale, target_mean, target_scale
    )
    val_dataset = GraphFileDataset(
        val_files, feature_mean, feature_scale, target_mean, target_scale
    )

    batch_size = int(train_cfg["batch_size"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = BAGNN(
        in_dim=int(model_cfg["input_dim"]),
        edge_dim=int(model_cfg["edge_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        num_tasks=len(model_cfg.get("outputs", ["Cp", "U", "V"])),
        dropout=float(model_cfg["dropout"]),
        heads=int(model_cfg["attention_heads"]),
    ).to(device)

    criterion = BoundaryWeightedMSE(
        beta=float(loss_cfg["boundary_weight_beta"]),
        task_weights=tuple(float(v) for v in loss_cfg.get("task_weights", [1.0, 1.0, 1.0])),
        normalization=str(loss_cfg.get("normalization", "sumW")),
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["initial_learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(train_cfg["scheduler_factor"]),
        patience=int(train_cfg["scheduler_patience"]),
        min_lr=float(train_cfg["minimum_learning_rate"]),
    )

    max_epochs = int(train_cfg["max_epochs"])
    grad_clip = float(train_cfg["gradient_clip_max_norm"])
    early_stop_patience = int(train_cfg["early_stopping_patience"])

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    history: List[Dict] = []

    best_path = output_dir / f"bagnn_seed{args.seed}_best.pt"
    history_path = output_dir / "training_history.csv"
    summary_path = output_dir / "run_summary.json"

    start = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, grad_clip
        )
        val_loss = validate_one_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)
        lr = float(optimizer.param_groups[0]["lr"])

        history.append(
            {
                "epoch": epoch,
                "train_optimization_loss_z": train_loss,
                "val_optimization_loss_z": val_loss,
                "learning_rate": lr,
            }
        )

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            no_improve = 0

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_name": "full_bagnn",
                "seed": int(args.seed),
                "best_epoch": int(best_epoch),
                "best_val_opt_loss_z": float(best_val),
                "in_dim": int(model_cfg["input_dim"]),
                "edge_dim": int(model_cfg["edge_dim"]),
                "hidden_dim": int(model_cfg["hidden_dim"]),
                "heads": int(model_cfg["attention_heads"]),
                "dropout": float(model_cfg["dropout"]),
                "beta": float(loss_cfg["boundary_weight_beta"]),
                "weighted_loss": 1,
                "loss_normalization": str(loss_cfg.get("normalization", "sumW")),
                "fixed_split_file": str(args.split),
                "normalization_file": str(args.normalization),
            }
            torch.save(checkpoint, best_path)
        else:
            no_improve += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == max_epochs:
            print(
                f"epoch={epoch:03d}/{max_epochs} | "
                f"train={train_loss:.6g} | "
                f"val={val_loss:.6g} | "
                f"best={best_val:.6g}@{best_epoch} | "
                f"lr={lr:.2e}"
            )

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(
                f"Early stopping at epoch {epoch}; "
                f"best checkpoint was epoch {best_epoch}."
            )
            break

    elapsed = time.perf_counter() - start
    write_history(history_path, history)

    summary = {
        "model": "full_bagnn",
        "seed": int(args.seed),
        "device": str(device),
        "train_cases": len(train_files),
        "validation_cases": len(val_files),
        "epochs_completed": len(history),
        "best_epoch": int(best_epoch),
        "best_validation_optimization_loss_z": float(best_val),
        "training_time_s": float(elapsed),
        "checkpoint": str(best_path),
        "training_history": str(history_path),
        "config": str(args.config),
        "fixed_split": str(args.split),
        "normalization": str(args.normalization),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 78)
    print("Training finished.")
    print(f"Best checkpoint : {best_path}")
    print(f"Training history: {history_path}")
    print(f"Run summary     : {summary_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
