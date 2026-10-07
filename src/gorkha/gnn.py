"""GNN on the ward graph.

A task is a set of nodes (a graph region), a set of visible labels in it (the survey), and a
set of target nodes. The model sees the features of all nodes and the visible labels. It
does not see a kriging estimate (decision of 2026-10-07).

The model has two parts:
- A distance layer gives each node a weighted mean of the visible labels. The weight of a
  surveyed ward decreases with its distance, and the model learns the ranges.
- Attention layers on the graph, with the edge length as an edge feature, calculate a
  correction from the features and the labels.

The model has two outputs for each node: the estimate and its variance. The loss is the
Gaussian negative log likelihood.
"""

import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv

from . import baselines

DEVICE = os.environ.get("GORKHA_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
EDGE_SCALE_KM = 10.0
# Start values of the ranges of the distance layer.
RANGES_KM = (5.0, 10.0, 20.0, 40.0)


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
                 dropout: float = 0.2, kernel: bool = True):
        super().__init__()
        self.kernel = kernel
        k = len(RANGES_KM) if kernel else 0
        self.log_range = nn.Parameter(torch.log(torch.tensor(RANGES_KM) / EDGE_SCALE_KM))
        self.mix = nn.Parameter(torch.zeros(len(RANGES_KM)))
        self.inp = nn.Linear(n_features + 2 + 2 * k, hidden)
        self.convs = nn.ModuleList(
            GATv2Conv(hidden, hidden // heads, heads=heads, edge_dim=1) for _ in range(layers))
        self.out = nn.Linear(hidden, 2)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, edge_index, edge_attr, dist, labels):
        base = 0.0
        if self.kernel:
            score = -dist[:, :, None] / torch.exp(self.log_range)   # nodes x surveyed x ranges
            local = (torch.softmax(score, dim=1) * labels[None, :, None]).sum(dim=1)
            support = torch.logsumexp(score, dim=1)
            x = torch.cat([x, local, support], dim=1)
            base = (local * torch.softmax(self.mix, dim=0)).sum(dim=1)
        h = torch.relu(self.inp(x))
        for conv in self.convs:
            h = h + torch.relu(conv(self.drop(h), edge_index, edge_attr))
        out = self.out(h)
        return base + out[:, 0], out[:, 1].clamp(-6, 4)


def _tensors(task: Task, x, y, pairs, xy):
    vis = task.visible.astype("float32")
    centre = float(y[task.nodes][task.visible].mean())
    spread = float(max(y[task.nodes][task.visible].var(), 1e-3))
    label = np.where(task.visible, y[task.nodes] - centre, 0.0)
    inp = np.column_stack([x[task.nodes], label, vis])
    p = induced_pairs(pairs, task.nodes, len(y))
    here = xy[task.nodes]
    length = np.linalg.norm(here[p[:, 0]] - here[p[:, 1]], axis=1)
    edges = np.concatenate([p, p[:, ::-1]]).T
    attr = np.concatenate([length, length])[:, None] / EDGE_SCALE_KM
    dist = np.linalg.norm(here[:, None, :] - here[task.visible][None, :, :], axis=2) / EDGE_SCALE_KM
    vis_labels = y[task.nodes][task.visible] - centre
    to = lambda a, t=torch.float32: torch.as_tensor(np.ascontiguousarray(a), dtype=t, device=DEVICE)
    return (to(inp), to(edges, torch.long), to(attr), to(dist), to(vis_labels), (centre, spread),
            to(y[task.nodes]), torch.as_tensor(task.target, device=DEVICE))


def train(tasks: list, x, y, pairs, xy, epochs: int, layers: int = 3, seed: int = 0,
          lr: float = 3e-3, weight_decay: float = 1e-3, kernel: bool = True) -> LabelGNN:
    torch.manual_seed(seed)
    data = [_tensors(t, x, y, pairs, xy) for t in tasks]
    model = LabelGNN(x.shape[1], layers=layers, kernel=kernel).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        inp, edges, attr, dist, labels, (centre, spread), truth, target = data[epoch % len(data)]
        if target.sum() == 0:
            continue
        delta, log_factor = model(inp, edges, attr, dist, labels)
        log_var = np.log(spread) + log_factor
        error = (centre + delta - truth) ** 2
        loss = (0.5 * (log_var + error / torch.exp(log_var)))[target].mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


@torch.no_grad()
def predict(model: LabelGNN, task: Task, x, y, pairs, xy) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and standard deviation for each node of the task."""
    model.eval()
    inp, edges, attr, dist, labels, (centre, spread), _, _ = _tensors(task, x, y, pairs, xy)
    delta, log_factor = model(inp, edges, attr, dist, labels)
    sd = torch.sqrt(spread * torch.exp(log_factor))
    return torch.clamp(centre + delta, 0, 1).cpu().numpy(), sd.cpu().numpy()


def fit_predict(x, xy, y, surveyed, pairs, seed: int = 0, members: int = 3, epochs: int = 200,
                n_tasks: int = 12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a group of models on the surveyed wards and estimate all wards.

    The training tasks hide 40% of the surveyed labels and ask the model for them.
    Return the estimate, the standard deviation, and the estimates of the members.
    The variance of the group is the mean of the member variances plus the variance of
    the member estimates.
    """
    n = len(y)
    rng = np.random.default_rng([7, seed])
    everything, tasks = np.arange(n), []
    for _ in range(n_tasks):
        hide = surveyed & (rng.random(n) < 0.4)
        if hide.sum() and (surveyed & ~hide).sum() >= 3:
            tasks.append(make_task(everything, surveyed & ~hide, hide))
    if not tasks:
        m = np.full(n, y[surveyed].mean())
        return m, np.full(n, y[surveyed].std()), m[None, :]
    test = make_task(everything, surveyed, ~surveyed)
    out = [predict(train(tasks, x, y, pairs, xy, epochs=epochs, seed=seed * 10 + k), test, x, y,
                   pairs, xy) for k in range(members)]
    means = np.array([o[0] for o in out])
    var = np.mean([o[1] ** 2 for o in out], axis=0) + means.var(axis=0)
    return means.mean(axis=0), np.sqrt(var), means
