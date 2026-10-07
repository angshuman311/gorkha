"""Pretrain the patch CNN without labels (contrastive learning) and write ward image vectors.

The CNN sees two changed copies of each Sentinel-2 patch and learns to give them similar
vectors, and different vectors to other patches. No damage label is used.
Output: data/processed/s2_embedding.parquet (one row for each ward).
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch import nn

from gorkha.paths import PROCESSED, ROOT

PATCHES = ROOT / "data" / "patches"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 128
N_COMPONENTS = 16


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class Encoder(nn.Module):
    """Small CNN: 6 x 64 x 64 patch to a vector of 128 numbers."""

    def __init__(self, channels: int = 6):
        super().__init__()
        self.body = nn.Sequential(block(channels, 32), block(32, 64), block(64, 96), block(96, DIM))
        self.head = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, 64))

    def features(self, x):
        return self.body(x).mean(dim=(2, 3))

    def forward(self, x):
        return F.normalize(self.head(self.features(x)), dim=1)


def to_float(x: torch.Tensor) -> torch.Tensor:
    """Reflectance values to a range near 0 to 1, with a compression of bright pixels."""
    return torch.log1p(x.float() / 1000.0) / 2.0


def augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    if torch.rand(1).item() < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))
    size = int(torch.randint(40, 65, (1,)))
    i, j = (int(torch.randint(0, 65 - size, (1,))) for _ in range(2))
    x = F.interpolate(x[:, :, i:i + size, j:j + size], size=64, mode="bilinear", align_corners=False)
    scale = 1 + 0.2 * (torch.rand(x.shape[0], 1, 1, 1, device=x.device) - 0.5)
    return x * scale + 0.02 * torch.randn_like(x)


def nt_xent(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z = torch.cat([a, b]).float()
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    n = a.shape[0]
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    torch.manual_seed(0)
    index = pd.read_parquet(PATCHES / "s2_index.parquet")
    arrays, offset, start = [], {}, 0
    for tile in sorted(index["tile"].unique()):
        a = np.load(PATCHES / f"s2_{tile}.npy")
        offset[tile] = start
        start += len(a)
        arrays.append(a)
    # The patches stay in the main memory. Only one batch at a time goes to the GPU.
    data = torch.from_numpy(np.minimum(np.concatenate(arrays), 32767).astype("int16"))
    row = index["tile"].map(offset).to_numpy() + index["i"].to_numpy()
    print(f"patches: {len(data)}, device: {DEVICE}", flush=True)

    model = Encoder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * (len(data) // 512)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(data))
        total = 0.0
        for k in range(len(data) // 512):
            x = to_float(data[perm[k * 512:(k + 1) * 512]].to(DEVICE))
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                za, zb = model(augment(x)), model(augment(x))
            loss = nt_xent(za.float(), zb.float())  # the loss needs full precision
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / (len(data) // 512):.3f}, {time.time() - t0:.0f} s", flush=True)
    torch.save(model.state_dict(), PROCESSED / "s2_encoder.pt")

    model.eval()
    feats = []
    with torch.no_grad():
        for k in range(0, len(data), 1024):
            feats.append(model.features(to_float(data[k:k + 1024].to(DEVICE))).float().cpu().numpy())
    feats = np.concatenate(feats)[row]
    wards = pd.read_parquet(PROCESSED / "wards.parquet", columns=["ward_id"])
    # Ward vector: mean of the patch vectors. PCA keeps the 16 strongest directions.
    mean = pd.DataFrame(feats).groupby(index["node"].to_numpy()).mean()
    comp = PCA(N_COMPONENTS, random_state=0).fit(mean.to_numpy())
    print("variance in the 16 components:", round(float(comp.explained_variance_ratio_.sum()), 3))
    z = comp.transform(mean.to_numpy())
    z = z / z.std(axis=0)
    out = pd.DataFrame(0.0, index=range(len(wards)), columns=[f"img_{k}" for k in range(N_COMPONENTS)])
    out.loc[mean.index] = z
    out["img_has"] = 0.0
    out.loc[mean.index, "img_has"] = 1.0
    out.insert(0, "ward_id", wards["ward_id"].to_numpy())
    out.to_parquet(PROCESSED / "s2_embedding.parquet", index=False)
    print(f"wards with an image vector: {int(out['img_has'].sum())} of {len(out)}")
