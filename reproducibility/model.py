# -*- coding: utf-8 -*-
"""Public BA-GNN model definition for checkpoint loading and inference.

This file contains the Full BA-GNN model architecture shared by the public
training, inference, and evaluation entry points.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax


def safe_torch_load(path: str, map_location="cpu"):
    """Load checkpoints across PyTorch versions with/without weights_only."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class BoundaryAttention(MessagePassing):
    """Residual boundary-aware multi-head attention used by Full BA-GNN."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 heads: int = 3, dropout: float = 0.1):
        super().__init__(aggr="add", node_dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout

        self.lin_key = nn.Linear(in_channels, heads * out_channels)
        self.lin_query = nn.Linear(in_channels, heads * out_channels)
        self.lin_value = nn.Linear(in_channels, heads * out_channels)
        self.edge_encoder = nn.Linear(edge_dim, heads * out_channels)
        self.boundary_attention = nn.Embedding(3, heads)
        self.lin_out = nn.Linear(heads * out_channels, out_channels)
        self.scale = out_channels ** 0.5

    def forward(self, x, edge_index, edge_attr, edge_type):
        identity = x
        query = self.lin_query(x).view(-1, self.heads, self.out_channels)
        key = self.lin_key(x).view(-1, self.heads, self.out_channels)
        value = self.lin_value(x).view(-1, self.heads, self.out_channels)
        edge_embedding = self.edge_encoder(edge_attr).view(-1, self.heads, self.out_channels)
        boundary_weights = self.boundary_attention(edge_type).unsqueeze(-1)

        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_embedding=edge_embedding,
            boundary_weights=boundary_weights,
        )
        out = self.lin_out(out.view(-1, self.heads * self.out_channels))
        return F.elu(identity + out)

    def message(self, query_i, key_j, value_j, edge_embedding,
                boundary_weights, index, ptr, size_i):
        scores = (query_i * key_j).sum(dim=-1) / self.scale
        edge_scores = (query_i * edge_embedding).sum(dim=-1) / self.scale
        final_scores = scores + edge_scores + boundary_weights.squeeze(-1)
        alpha = softmax(final_scores, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return value_j * alpha.unsqueeze(-1)


class EdgeGCNConv(MessagePassing):
    """Residual edge-aware GCN layer used by Full BA-GNN."""

    def __init__(self, in_channels: int, out_channels: int, edge_dim: int,
                 dropout: float = 0.1):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels)
        self.edge_encoder = nn.Linear(edge_dim, out_channels)
        self.node_type_encoder = nn.Embedding(3, out_channels)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, node_type):
        num_nodes = x.size(0)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        num_added = edge_index.size(1) - edge_attr.size(0)
        if num_added > 0:
            loop_attr = edge_attr.new_zeros((num_added, edge_attr.size(1)))
            edge_attr = torch.cat([edge_attr, loop_attr], dim=0)

        row, col = edge_index
        deg = degree(row, num_nodes, dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0.0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        identity = self.lin(x)
        edge_embedding = self.edge_encoder(edge_attr)
        neighbor_type_emb = self.node_type_encoder(node_type)[col]

        out = self.propagate(
            edge_index,
            x=identity,
            norm=norm,
            edge_embedding=edge_embedding,
            neighbor_type_emb=neighbor_type_emb,
        )
        out = identity + out
        out = self.norm(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        return out

    def message(self, x_j, norm, edge_embedding, neighbor_type_emb):
        message = x_j + edge_embedding + neighbor_type_emb
        return norm.view(-1, 1) * message


class BAGNN(nn.Module):
    """Full BA-GNN architecture corresponding to the released checkpoint."""

    def __init__(self, in_dim: int, edge_dim: int, hidden_dim: int = 128,
                 num_tasks: int = 3, dropout: float = 0.1, heads: int = 3):
        super().__init__()
        self.dropout = dropout
        self.input_proj = nn.Linear(in_dim, hidden_dim)

        self.boundary_attention = BoundaryAttention(
            hidden_dim, hidden_dim, edge_dim, heads=heads, dropout=dropout
        )
        self.final_attention = BoundaryAttention(
            hidden_dim, hidden_dim, edge_dim, heads=heads, dropout=dropout
        )

        self.enc_conv1 = EdgeGCNConv(hidden_dim, hidden_dim, edge_dim, dropout)
        self.enc_conv2 = EdgeGCNConv(hidden_dim, hidden_dim, edge_dim, dropout)
        self.enc_conv3 = EdgeGCNConv(hidden_dim, hidden_dim, edge_dim, dropout)

        self.original_proj = nn.Linear(in_dim, hidden_dim)
        self.dec_conv1 = EdgeGCNConv(hidden_dim * 2, hidden_dim, edge_dim, dropout)
        self.dec_conv2 = EdgeGCNConv(hidden_dim * 2, hidden_dim, edge_dim, dropout)
        self.dec_conv3 = EdgeGCNConv(hidden_dim * 2, hidden_dim, edge_dim, dropout)

        self.task_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
            for _ in range(num_tasks)
        ])

    def forward(self, data: Data):
        x0 = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        node_type = data.node_type
        edge_type = data.edge_type

        x = F.elu(self.input_proj(x0))
        x = self.boundary_attention(x, edge_index, edge_attr, edge_type)

        x1 = F.elu(self.enc_conv1(x, edge_index, edge_attr, node_type))
        x2 = F.elu(self.enc_conv2(x1, edge_index, edge_attr, node_type))
        x3 = F.elu(self.enc_conv3(x2, edge_index, edge_attr, node_type))

        x1_up = F.elu(self.dec_conv1(torch.cat([x3, x2], dim=1), edge_index, edge_attr, node_type))
        x2_up = F.elu(self.dec_conv2(torch.cat([x1_up, x1], dim=1), edge_index, edge_attr, node_type))

        original_projected = F.elu(self.original_proj(x0))
        x3_up = F.elu(self.dec_conv3(
            torch.cat([x2_up, original_projected], dim=1),
            edge_index, edge_attr, node_type
        ))

        shared = self.final_attention(x3_up, edge_index, edge_attr, edge_type)
        outs = [head(shared) for head in self.task_heads]
        return torch.cat(outs, dim=1)


def build_bagnn_from_checkpoint(checkpoint: dict, device="cpu") -> BAGNN:
    """Construct Full BA-GNN using architecture values stored in checkpoint."""
    model_name = str(checkpoint.get("model_name", "full_bagnn")).lower()
    if model_name not in {"full_bagnn", "bagnn", "full", "g_full"}:
        raise ValueError(f"Checkpoint is not a Full BA-GNN checkpoint: {model_name}")

    model = BAGNN(
        in_dim=int(checkpoint["in_dim"]),
        edge_dim=int(checkpoint["edge_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        num_tasks=3,
        dropout=float(checkpoint.get("dropout", 0.1)),
        heads=int(checkpoint.get("heads", 3)),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model
