"""
Two models for the same node-level task, differing only in whether they are
allowed to look at the graph.

CascadeGNN  - GraphSAGE layers. Each layer lets a node update its own state
              using the states of its neighbours, so after k layers a node has
              "seen" everything within k hops. That is the inductive bias we
              want: failure propagates along the network, so a node's outcome
              should depend on its neighbourhood, not just on how deep the
              water is at its own site.

NodeMLP     - identical inputs, identical depth and width, no message passing.
              Every node is classified in isolation.

The comparison between them is the experiment. If the GNN wins, the graph
structure is carrying information that per-site hazard intensity does not,
which is the whole premise of modelling cascading failure as a network
problem rather than a site-by-site one.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class CascadeGNN(nn.Module):
    """GraphSAGE node classifier. Output is one logit per node."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.convs.append(SAGEConv(d, hidden))
            self.norms.append(nn.LayerNorm(hidden))
            d = hidden
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)


class NodeMLP(nn.Module):
    """Same shape, no neighbours. The control condition."""

    def __init__(self, in_dim: int, hidden: int = 64, layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.dropout = dropout
        self.lins = nn.ModuleList()
        self.norms = nn.ModuleList()
        d = in_dim
        for _ in range(layers):
            self.lins.append(nn.Linear(d, hidden))
            self.norms.append(nn.LayerNorm(hidden))
            d = hidden
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor | None = None) -> torch.Tensor:
        for lin, norm in zip(self.lins, self.norms):
            x = lin(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.head(x).squeeze(-1)


def build(name: str, in_dim: int, **kw) -> nn.Module:
    return {"gnn": CascadeGNN, "mlp": NodeMLP}[name](in_dim, **kw)
