"""GNN on the ward graph (rung 3 of the ladder).

A task is a set of nodes (a graph region), a set of visible labels in it (the survey), and a
set of target nodes. The model sees the features of all nodes and the visible labels. It
does not see a kriging estimate (decision of 2026-10-07). Attention layers with the edge
length as an edge feature pass the visible labels through the graph.

The output is a difference from the mean of the visible labels. The last layer starts at
zero, thus the model starts as "mean of the surveyed wards".
"""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from . import baselines

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EDGE_SCALE_KM = 10.0


@dataclass
class Task:
    nodes: np.ndarray      # node indices of the region (in the full graph)
    visible: np.ndarray    # boolean on `nodes`: labels that the model sees
    target: np.ndarray     # boolean on `nodes`: nodes in the loss or in the score
    ok: np.ndarray | None = None  # ordinary kriging estimate, only as a reference


def induced_pairs(pairs: np.ndarray, nodes: np.ndarray, n: int) -> np.ndarray:
    """Edges with the two ends in `nodes`, with indices of the region."""
    new = np.full(n, -1)
    new[nodes] = np.arange(len(nodes))
    p = new[pairs]
    return p[(p >= 0).all(axis=1)]


def make_task(nodes, visible, target, xy=None, y=None) -> Task:
    """Give `xy` and `y` to add the ordinary kriging reference to the task."""
    ok = None
    if xy is not None:
        z, _ = baselines.krige(xy[nodes][visible], y[nodes][visible], xy[nodes])
        ok = np.clip(z, 0, 1)
    return Task(nodes, visible, target, ok)


class LabelGNN(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32, layers: int = 3, heads: int = 4,
                 dropout: float = 0.2):
        super().__init__()
        self.inp = nn.Linear(n_features + 2, hidden)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=1) for _ in range(layers))
        self.out = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.inp(x))
        for conv in self.convs:
            h = h + torch.relu(conv(self.drop(h), edge_index, edge_attr))
        return self.out(h).squeeze(-1)


def _tensors(task: Task, x, y, pairs, xy):
    vis = task.visible.astype("float32")
    centre = float(y[task.nodes][task.visible].mean())
    label = np.where(task.visible, y[task.nodes] - centre, 0.0)
    inp = np.column_stack([x[task.nodes], label, vis])
    p = induced_pairs(pairs, task.nodes, len(y))
    length = np.linalg.norm(xy[task.nodes][p[:, 0]] - xy[task.nodes][p[:, 1]], axis=1)
    edges = np.concatenate([p, p[:, ::-1]]).T
    attr = np.concatenate([length, length])[:, None] / EDGE_SCALE_KM
    to = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device=DEVICE)
    return (to(inp), to(edges, torch.long), to(attr), centre, to(y[task.nodes]),
            torch.as_tensor(task.target, device=DEVICE))


def train(tasks: list, x, y, pairs, xy, epochs: int, layers: int = 3, seed: int = 0,
          lr: float = 3e-3, weight_decay: float = 1e-3) -> LabelGNN:
    torch.manual_seed(seed)
    data = [_tensors(t, x, y, pairs, xy) for t in tasks]
    model = LabelGNN(x.shape[1], layers=layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        inp, edges, attr, centre, truth, target = data[epoch % len(data)]
        if target.sum() == 0:
            continue
        pred = centre + model(inp, edges, attr)
        loss = ((pred - truth)[target] ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict(model: LabelGNN, task: Task, x, y, pairs, xy) -> np.ndarray:
    model.eval()
    inp, edges, attr, centre, _, _ = _tensors(task, x, y, pairs, xy)
    return torch.clamp(centre + model(inp, edges, attr), 0, 1).cpu().numpy()
