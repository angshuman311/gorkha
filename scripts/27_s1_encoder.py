"""Pretrain a small CNN on the Sentinel-1 coherence patches without labels (mode 1), and
write one image vector for each ward.

The patches have 5 channels (coherence before, across the first earthquake, across the
second earthquake, amplitude before, amplitude after) at 40 m, 32 x 32 pixels. The SSL4EO
Sentinel-1 encoder does not know coherence, so this encoder is our own, with the same
contrastive recipe as the Sentinel-2 encoder (scripts/19_cnn_pretrain.py).
Output: data/processed/s1_embedding.parquet
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
DIM = 96
N_COMPONENTS = 16


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
                         nn.ReLU(inplace=True), nn.Conv2d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True), nn.MaxPool2d(2))


class Encoder(nn.Module):
    def __init__(self, channels: int = 5):
        super().__init__()
        self.body = nn.Sequential(block(channels, 32), block(32, 64), block(64, DIM))
        self.head = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, 48))

    def features(self, x):
        return self.body(x).mean(dim=(2, 3))

    def forward(self, x):
        return F.normalize(self.head(self.features(x)), dim=1)


def augment(x: torch.Tensor) -> torch.Tensor:
    if torch.rand(1).item() < 0.5:
        x = x.flip(3)
    if torch.rand(1).item() < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(torch.randint(0, 4, (1,))), (2, 3))
    size = int(torch.randint(22, 33, (1,)))
    i, j = (int(torch.randint(0, 33 - size, (1,))) for _ in range(2))
    x = F.interpolate(x[:, :, i:i + size, j:j + size], size=32, mode="bilinear", align_corners=False)
    return x + 0.03 * torch.randn_like(x)


def nt_xent(a, b, temperature: float = 0.2):
    z = torch.cat([a, b]).float()
    sim = z @ z.T / temperature
    sim.fill_diagonal_(-1e9)
    n = a.shape[0]
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    args = parser.parse_args()
    torch.manual_seed(0)
    index = pd.read_parquet(PATCHES / "s1_index.parquet")
    data = torch.from_numpy(np.load(PATCHES / "s1_patches.npy").astype("float32"))
    # Standardize each channel with the values of all patches.
    mean = data.mean(dim=(0, 2, 3), keepdim=True)
    sd = data.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    data = (data - mean) / sd
    print(f"patches: {len(data)}, device: {DEVICE}", flush=True)
    model = Encoder().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    batch = 256
    steps = args.epochs * (len(data) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=max(steps, 1))
    t0 = time.time()
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(data))
        total = 0.0
        for k in range(len(data) // batch):
            x = data[perm[k * batch:(k + 1) * batch]].to(DEVICE)
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                za, zb = model(augment(x)), model(augment(x))
            loss = nt_xent(za, zb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / max(len(data) // batch, 1):.3f}, {time.time() - t0:.0f} s",
              flush=True)
    torch.save(model.state_dict(), PROCESSED / "s1_encoder.pt")

    model.eval()
    feats = []
    with torch.no_grad():
        for k in range(0, len(data), 1024):
            feats.append(model.features(data[k:k + 1024].to(DEVICE)).float().cpu().numpy())
    feats = np.concatenate(feats)
    wards = pd.read_parquet(PROCESSED / "wards.parquet", columns=["ward_id"])
    per_ward = pd.DataFrame(feats).groupby(index["node"].to_numpy()).mean()
    comp = PCA(N_COMPONENTS, random_state=0).fit(per_ward.to_numpy())
    z = comp.transform(per_ward.to_numpy())
    z = z / z.std(axis=0)
    out = pd.DataFrame(0.0, index=range(len(wards)), columns=[f"img_{k}" for k in range(N_COMPONENTS)])
    out.loc[per_ward.index] = z
    out["img_has"] = 0.0
    out.loc[per_ward.index, "img_has"] = 1.0
    out.insert(0, "ward_id", wards["ward_id"].to_numpy())
    out.to_parquet(PROCESSED / "s1_embedding.parquet", index=False)
    print(f"wards with an image vector: {int(out['img_has'].sum())} of {len(out)}, "
          f"variance in the {N_COMPONENTS} components: {comp.explained_variance_ratio_.sum():.3f}")
